#ifndef IMU_DRIVER_HPP
#define IMU_DRIVER_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>


// ===================================================================
// 亚博智能 (Yahboom) / WitMotion 兼容 AHRS IMU UART 驱动
//
// 协议 A：WitMotion/JY/Yahboom-10axis 常见 0x55 帧：
//   - 每帧固定 11 字节： [0x55] [type] [payload×8] [checksum]
//   - type 0x51: 加速度  AX/AY/AZ + Temp     (int16 little-endian)
//   - type 0x52: 角速度  GX/GY/GZ + Voltage
//   - type 0x53: 欧拉角  Roll/Pitch/Yaw + Version
//   - type 0x54: 磁力计  MX/MY/MZ + Temp      (当前 API 不暴露)
//   - type 0x59: 四元数  q0/q1/q2/q3
//   - checksum = (sum of bytes [0..9]) & 0xFF
//
//   物理量换算（手册）：
//     accel  = raw / 32768 * 16g          (g = 9.80665 m/s^2)
//     gyro   = raw / 32768 * 2000 deg/s   (再转 rad/s)
//     angle  = raw / 32768 * 180 deg
//     quat   = raw / 32768
//
// 协议 B：Yahboom 9-axis / FDILink 常见 0xFC 帧：
//   - [0xFC] [type] [payload_len] [seq] [header_crc8]
//     [payload_crc16_hi] [payload_crc16_lo] [payload] [0xFD]
//   - type 0x40 MSG_IMU  payload 56 bytes，float32 LE：gyro rad/s,
//     accel m/s^2, mag, temp, pressure, timestamp
//   - type 0x41 MSG_AHRS payload 48 bytes，float32 LE：Euler rad,
//     quaternion Q1..Q4, timestamp
//
// 10 轴常见出厂波特率 9600bps；9 轴 FDILink 常见出厂波特率 921600bps。
// 实际设备仍以厂家上位机配置为准。
//
// 注意：IMU 与电机不共享串口。两路都通过 USB 转串口接入：
//       电机走 USB 转 485（例如 /dev/ttyUSB0），
//       IMU  走 USB 转 TTL（例如 /dev/ttyUSB1）。
//       实际节点统一由 robot_config.yaml::devices 配置，
//       建议用 udev 规则按物理端口固定别名（如 /dev/imu /dev/motor）。
// ===================================================================


class IMUDriver
{
public:
    explicit IMUDriver(std::string port = "/dev/ttyUSB1",
                       int baud = 9600,
                       int health_timeout_ms = 50);
    ~IMUDriver();

    // 禁止拷贝（持有文件描述符和后台线程）
    IMUDriver(const IMUDriver &)            = delete;
    IMUDriver &operator=(const IMUDriver &) = delete;

    // 打开串口、配置 termios、启动后台读线程
    bool Start();

    // 通知读线程退出，关闭串口
    void Stop();

    // 拷贝出最新可用快照（线程安全）
    // 返回 false 表示姿态、gyro、accel 任一路未收到或已超过 health_timeout_ms
    bool Snapshot(std::array<float, 4> &quat_wxyz,
                  std::array<float, 3> &gyro,
                  std::array<float, 3> &accel) const;

    // 姿态、gyro、accel 三路数据是否都新鲜
    bool IsHealthy() const;

    // 最近一次姿态更新的时间（单调时钟，微秒）
    uint64_t LastUpdateUs() const;

    // 最近一次串口错误（用于上层日志）
    std::string LastError() const;

#ifdef IMU_DRIVER_ENABLE_TEST_FEED
    // 测试专用：绕过串口，把字节直接喂给生产解析状态机。
    void DebugFeedBytesForTest(const uint8_t *data, std::size_t len);

    // 测试专用：验证 select/FD_SET 的 fd 范围防线。
    static bool DebugIsSelectableFdForTest(int fd);
#endif

private:
    // 后台读线程主体
    void ReaderLoop();

    // 字节级状态机：支持 0x55 固定帧和 0xFC FDILink 帧。
    void HandleByte(uint8_t byte);

    // 拆出来的解码函数（按 type 分发到不同字段）。
    void DecodeWitFrame(uint8_t type, const uint8_t payload[8]);
    void DecodeFdiFrame(uint8_t type, const uint8_t *payload, std::size_t len);

    void StoreQuaternion(std::array<float, 4> quat_wxyz);
    void StoreEulerRadians(float roll, float pitch, float yaw);
    void ResetParser();
    void InvalidateSamples();
    void RecordProtocolError(const char *message);
    void RecordReaderFailure(const char *operation, int error_number);
    bool SamplesFreshLocked(uint64_t now_us) const;

    // 配置 termios 串口参数
    bool ConfigureTermios();

    // 把波特率整数映射到 termios 的 speed_t（B9600 / B115200 / B921600 ...）
    static unsigned int MapBaud(int baud);

private:
    std::string port_;
    int         baud_;
    int         health_timeout_ms_;
    int         fd_ = -1;

    std::thread       reader_thread_;
    std::atomic<bool> running_{false};

    // 帧解析状态机内部缓冲（仅 ReaderLoop 线程使用，无锁）
    enum class ParseState
    {
        WaitHeader,
        WitReadType,
        WitReadPayload,
        WitReadChecksum,
        FdiReadType,
        FdiReadLength,
        FdiReadSequence,
        FdiReadHeaderCrc,
        FdiReadDataCrcHigh,
        FdiReadDataCrcLow,
        FdiReadPayload,
        FdiReadEnd
    };

    static constexpr std::size_t kMaxPayloadBytes = 56U;

    ParseState parse_state_ = ParseState::WaitHeader;
    uint8_t    type_byte_   = 0;
    uint8_t    payload_buf_[kMaxPayloadBytes]{};
    std::size_t payload_idx_ = 0U;
    uint8_t    payload_len_ = 0U;
    uint16_t   running_sum_ = 0; // 累积校验和
    uint8_t    fdi_header_buf_[4]{};
    uint16_t   fdi_expected_crc16_ = 0U;

    // 最新快照（被 ReaderLoop 写，被 Snapshot() 读）。
    // 三路数据分别跟踪有效位和时间戳，避免默认值或陈旧 gyro/accel 被当成安全快照。
    mutable std::mutex     sample_mutex_;
    std::array<float, 4>   latest_quat_{1.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 3>   latest_gyro_{0.0f, 0.0f, 0.0f};
    std::array<float, 3>   latest_accel_{0.0f, 0.0f, 0.0f};
    uint64_t               orientation_update_us_ = 0U;
    uint64_t               gyro_update_us_ = 0U;
    uint64_t               accel_update_us_ = 0U;
    bool                   orientation_valid_ = false;
    bool                   gyro_valid_ = false;
    bool                   accel_valid_ = false;

    mutable std::mutex last_error_mutex_;
    std::string        last_error_;
};


#endif // IMU_DRIVER_HPP
