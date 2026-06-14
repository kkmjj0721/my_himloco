#include "imu_driver.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <utility>


namespace 
{

constexpr float kDegToRad       = 3.14159265358979323846f / 180.0f;     // 角度转弧度
constexpr float kAccelScale_g   = 16.0f;                                // 加速度计量程，+/-16g range
constexpr float kGyroScale_dps  = 2000.0f;                              // 陀螺仪量程，+/-2000 deg/s range
constexpr float kAngleScale_deg = 180.0f;                               // 角度比例尺，+/-180 deg
constexpr float kGravity_mps2   = 9.80665f;                             // 重力加速度
constexpr float kMaxSafeGyroRadps  = 4000.0f * kDegToRad;               // 最大允许角速度
constexpr float kMaxSafeAccelMps2  = 64.0f * kGravity_mps2;             // 最大允许加速度

constexpr uint8_t kWitHeader    = 0x55U;        // 帧头
constexpr uint8_t kWitTypeAccel = 0x51U;        // 加速度标识
constexpr uint8_t kWitTypeGyro  = 0x52U;        // 角速度标识
constexpr uint8_t kWitTypeAngle = 0x53U;        // 角度标识
constexpr uint8_t kWitTypeMag   = 0x54U;        // 磁场标识
constexpr uint8_t kWitTypeQuat  = 0x59U;        // 四元数标识

// --- 节点健康检查的时间阈值限制 ---
constexpr int kMinHealthTimeoutMs = 1;
constexpr int kMaxHealthTimeoutMs = 60000;

// --- 字节重组函数：小端模式解析 (Little-Endian) ---

// 将两个 8 位无符号整数拼接为一个 16 位有符号整数
inline int16_t MakeInt16LE(uint8_t lo, uint8_t hi)
{
    // 将高位左移 8 位，然后与低位按位或 (OR)
    return static_cast<int16_t>((static_cast<uint16_t>(hi) << 8U) |
                                static_cast<uint16_t>(lo));
}

// 将四个 8 位字节拼接为一个 32 位无符号整数 (小端)
inline uint32_t MakeUInt32LE(const uint8_t *data)
{
    return (static_cast<uint32_t>(data[0])      ) |
           (static_cast<uint32_t>(data[1]) << 8U ) |
           (static_cast<uint32_t>(data[2]) << 16U) |
           (static_cast<uint32_t>(data[3]) << 24U);
}

// 将四个字节解析为 IEEE-754 标准的单精度浮点数
float MakeFloatLE(const uint8_t *data)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be IEEE-754 32-bit");
    const uint32_t raw = MakeUInt32LE(data);
    float value = 0.0f;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

// --- 数学工具与合法性检查 ---

// 归一化四元数 (确保 w^2 + x^2 + y^2 + z^2 = 1)
// 如果输入的四元数无效（包含 NaN 或是全0），返回 false
bool NormalizeQuaternion(std::array<float, 4> *quat)
{
    float norm_sq = 0.0f;           // 归一化参数

    for (float v : *quat)           
    {
        if (!std::isfinite(v))  return false;       // 检查是否为无限大或 NaN
        norm_sq += v * v;
    }

    if ((norm_sq <= 1.0e-12f) || !std::isfinite(norm_sq))   return false;       // 避免除以 0

    const float inv_norm = 1.0f / std::sqrt(norm_sq);           // 乘以倒数比直接除法效率高
    for (float &v : *quat)
    {
        v *= inv_norm;
    }
    return true;
}

// 检查文件描述符是否在 select() 允许的范围内 (通常是 0~1023)
bool IsSelectableFd(int fd)
{
    return (fd >= 0) && (fd < FD_SETSIZE);
}

// 检查三轴向量是否超出了设定的物理极值 (防止脏数据导致控制算法崩溃)
bool VectorWithinAbsLimit(const std::array<float, 3> &values, float max_abs)
{
    for (float value : values)
    {
        if (!std::isfinite(value) || (std::fabs(value) > max_abs))
        {
            return false;
        }
    }
    return true;
}

// 获取当前系统时间的微秒时间戳 (单调时钟，不受系统修改时间影响)
uint64_t NowMonoUs()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

// 限制健康超时的范围 (防止配置为 0 或极大值)
int ClampHealthTimeoutMs(int timeout_ms)
{
    if (timeout_ms < kMinHealthTimeoutMs)
    {
        return kMinHealthTimeoutMs;
    }
    if (timeout_ms > kMaxHealthTimeoutMs)
    {
        return kMaxHealthTimeoutMs;
    }
    return timeout_ms;
}

} // namespace


// 构造函数：初始化串口名、波特率和超时阈值
IMUDriver::IMUDriver(std::string port, int baud, int health_timeout_ms)
    : port_(std::move(port)),
      baud_(baud),
      health_timeout_ms_(ClampHealthTimeoutMs(health_timeout_ms))
{
}

// 析构函数：确保对象销毁时，停止后台线程并关闭串口
IMUDriver::~IMUDriver()
{
    Stop();
}

// 启动驱动
bool IMUDriver::Start()
{
    if (running_)   return true;            // 防止重复启动
    if (reader_thread_.joinable())  reader_thread_.join();      // 回收残存线程

    // 清理旧句柄
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }

    // 启动前先将所有旧数据标记为无效
    InvalidateSamples();

    // 以读写(O_RDWR)、不作为控制终端(O_NOCTTY)、非阻塞(O_NONBLOCK)的方式打开串口
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0)
    {
        std::lock_guard<std::mutex> lock(last_error_mutex_);
        last_error_ = "open(" + port_ + ") failed: " + std::strerror(errno);
        return false;
    }

    // POSIX select 机制对 fd 的大小有限制，需校验
    if (!IsSelectableFd(fd_))
    {
        {
            std::lock_guard<std::mutex> lock(last_error_mutex_);
            last_error_ = "open(" + port_ + ") returned fd outside select range: " +
                          std::to_string(fd_);
        }
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // 配置串口参数 (波特率、停止位、校验等)
    if (!ConfigureTermios())
    {
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // 重置解析状态机
    ResetParser();
    {
        std::lock_guard<std::mutex> lock(last_error_mutex_);
        last_error_.clear();
    }

    // 设置运行标志位，并启动后台读取线程
    running_ = true;
    reader_thread_ = std::thread(&IMUDriver::ReaderLoop, this);
    return true;
}

// 停止驱动
void IMUDriver::Stop()
{
    running_.store(false);      // 通知线程准备退出

    if (reader_thread_.joinable())
    {
        reader_thread_.join();  // 阻塞等待后台线程安全结束
    }
    if (fd_ >= 0)
    {
        ::close(fd_);   // 关闭串口
        fd_ = -1;
    }
    InvalidateSamples();        // 将数据置为无效
}

// 对外提供最新数据的“快照”
bool IMUDriver::Snapshot(std::array<float, 4> &quat_wxyz,
                         std::array<float, 3> &gyro,
                         std::array<float, 3> &accel) const
{
    std::lock_guard<std::mutex> lock(sample_mutex_);            // 加锁，防止读取一半时后台线程正在写
    if (!SamplesFreshLocked(NowMonoUs()))                       // 检查数据是否在超时阈值内 (新鲜度检查)
    {
        return false;
    }
    // 拷贝数据给外部引用
    quat_wxyz = latest_quat_;
    gyro      = latest_gyro_;
    accel     = latest_accel_;
    return true;
}

// 检查驱动是否健康 (数据没有断更)
bool IMUDriver::IsHealthy() const
{
    std::lock_guard<std::mutex> lock(sample_mutex_);
    return SamplesFreshLocked(NowMonoUs());
}


uint64_t IMUDriver::LastUpdateUs() const
{
    std::lock_guard<std::mutex> lock(sample_mutex_);
    return orientation_update_us_;
}


std::string IMUDriver::LastError() const
{
    std::lock_guard<std::mutex> lock(last_error_mutex_);
    return last_error_;
}


#ifdef IMU_DRIVER_ENABLE_TEST_FEED
void IMUDriver::DebugFeedBytesForTest(const uint8_t *data, std::size_t len)
{
    if ((data == nullptr) && (len > 0U))
    {
        return;
    }
    for (std::size_t i = 0U; i < len; ++i)
    {
        HandleByte(data[i]);
    }
}


bool IMUDriver::DebugIsSelectableFdForTest(int fd)
{
    return IsSelectableFd(fd);
}
#endif

// 配置 termios (Linux/POSIX 下串口的标准配置结构体)
bool IMUDriver::ConfigureTermios()
{
    termios tio{};
    // 获取当前串口配置
    if (tcgetattr(fd_, &tio) != 0)
    {
        std::lock_guard<std::mutex> lock(last_error_mutex_);
        last_error_ = std::string("tcgetattr: ") + std::strerror(errno);
        return false;
    }

    cfmakeraw(&tio); // 将串口设置为“原始模式”(Raw Mode)，禁用所有回车换行等特殊字符的翻译

    const speed_t speed = MapBaud(baud_); // 将整数波特率映射为 POSIX 常量
    cfsetispeed(&tio, speed); // 设置输入波特率
    cfsetospeed(&tio, speed); // 设置输出波特率

    tio.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态线，使能接收器
    tio.c_cflag &= ~CSIZE;           // 清除当前字符大小掩码
    tio.c_cflag |= CS8;              // 数据位：8位
    tio.c_cflag &= ~PARENB;          // 校验位：无 (No Parity)
    tio.c_cflag &= ~CSTOPB;          // 停止位：1个 (清除两位停止位的标志)
    tio.c_cflag &= ~CRTSCTS;         // 禁用硬件流控

    tio.c_iflag &= ~(IXON | IXOFF | IXANY); // 禁用软件流控 (XON/XOFF)
    tio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 再次确保非规范模式、关闭回显和信号生成
    tio.c_oflag &= ~OPOST;           // 禁用输出处理

    // VMIN 和 VTIME 在非阻塞模式(O_NONBLOCK)下一般被忽略，但显式设为 0 是好习惯
    tio.c_cc[VMIN]  = 0;
    tio.c_cc[VTIME] = 0;

    tcflush(fd_, TCIFLUSH); // 刷新（清空）输入缓冲区里陈旧的垃圾数据
    // 立即应用配置 (TCSANOW)
    if (tcsetattr(fd_, TCSANOW, &tio) != 0)
    {
        std::lock_guard<std::mutex> lock(last_error_mutex_);
        last_error_ = std::string("tcsetattr: ") + std::strerror(errno);
        return false;
    }
    return true;
}


unsigned int IMUDriver::MapBaud(int baud)
{
    switch (baud)
    {
        case 9600:    return B9600;
        case 19200:   return B19200;
        case 38400:   return B38400;
        case 57600:   return B57600;
        case 115200:  return B115200;
        case 230400:  return B230400;
        case 460800:  return B460800;
        case 921600:  return B921600;
        default:      return B9600;
    }
}

// 核心后台读取循环
void IMUDriver::ReaderLoop()
{
    uint8_t buf[64]; // 一次最多读取的字节数
    while (running_) // 只要没有被停止，就一直循环
    {
        if (!IsSelectableFd(fd_)) { /* 异常处理略 */ break; }

        fd_set rset;      // 定义文件描述符集合
        FD_ZERO(&rset);   // 清空集合
        FD_SET(fd_, &rset); // 将当前串口句柄加入集合
        timeval tv{0, 20000}; // 设置 select 超时时间为 20 毫秒

        // 阻塞等待串口有数据到来，或者超时。
        // 第一个参数必须是最大 fd 值 + 1
        const int ret = select(fd_ + 1, &rset, nullptr, nullptr, &tv);
        if (ret < 0) // 发生错误
        {
            if (errno == EINTR) continue; // 如果是被系统信号中断 (比如 Ctrl+C)，则继续循环
            RecordReaderFailure("select", errno);
            break;
        }
        if (ret == 0) continue; // 超时了，没数据，直接进入下一轮循环

        // select 告诉我们有数据了，去读
        const ssize_t n = ::read(fd_, buf, sizeof(buf));
        if (n < 0)
        {
            // 非阻塞模式下，如果没有数据读会返回 EAGAIN，忽略即可
            if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) continue;
            RecordReaderFailure("read", errno);
            break;
        }
        if (n == 0) // 读到 0 字节，通常代表串口物理断开 (EOF)
        {
            RecordReaderFailure("read EOF", 0);
            break;
        }
        
        // 读到数据了，逐个字节喂给状态机解析
        for (ssize_t i = 0; i < n; ++i)
        {
            HandleByte(buf[i]);
        }
    }
}


void IMUDriver::InvalidateSamples()
{
    std::lock_guard<std::mutex> lock(sample_mutex_);
    latest_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};
    latest_gyro_ = {0.0f, 0.0f, 0.0f};
    latest_accel_ = {0.0f, 0.0f, 0.0f};
    orientation_update_us_ = 0U;
    gyro_update_us_ = 0U;
    accel_update_us_ = 0U;
    orientation_valid_ = false;
    gyro_valid_ = false;
    accel_valid_ = false;
}


void IMUDriver::RecordProtocolError(const char *message)
{
    std::lock_guard<std::mutex> lock(last_error_mutex_);
    last_error_ = (message != nullptr) ? message : "IMU protocol error";
}


void IMUDriver::RecordReaderFailure(const char *operation, int error_number)
{
    {
        std::lock_guard<std::mutex> lock(last_error_mutex_);
        if (error_number == 0)
        {
            last_error_ = operation;
        }
        else
        {
            last_error_ = std::string(operation) + ": " + std::strerror(error_number);
        }
    }
    InvalidateSamples();
    running_.store(false);
}


bool IMUDriver::SamplesFreshLocked(uint64_t now_us) const
{
    if (!orientation_valid_ || !gyro_valid_ || !accel_valid_)
    {
        return false;
    }

    const uint64_t timeout_us = static_cast<uint64_t>(health_timeout_ms_) * 1000ULL;
    const auto fresh = [now_us, timeout_us](uint64_t sample_us) -> bool
    {
        return (sample_us > 0U) &&
               (now_us >= sample_us) &&
               ((now_us - sample_us) < timeout_us);
    };

    return fresh(orientation_update_us_) &&
           fresh(gyro_update_us_) &&
           fresh(accel_update_us_);
}


void IMUDriver::ResetParser()
{
    parse_state_ = ParseState::WaitHeader;
    type_byte_ = 0U;
    payload_idx_ = 0U;
    payload_len_ = 0U;
    running_sum_ = 0U;
    fdi_expected_crc16_ = 0U;
}


void IMUDriver::HandleByte(uint8_t byte)
{
    switch (parse_state_)
    {
        case ParseState::WaitHeader:
            if (byte == kWitHeader)
            {
                running_sum_ = byte;
                payload_idx_ = 0U;
                parse_state_ = ParseState::WitReadType;
            }
            return;

        case ParseState::WitReadType:
            type_byte_ = byte;
            running_sum_ = static_cast<uint16_t>(running_sum_ + byte);
            payload_idx_ = 0U;
            parse_state_ = ParseState::WitReadPayload;
            return;

        case ParseState::WitReadPayload:
            payload_buf_[payload_idx_] = byte;
            ++payload_idx_;
            running_sum_ = static_cast<uint16_t>(running_sum_ + byte);
            if (payload_idx_ >= 8U)
            {
                parse_state_ = ParseState::WitReadChecksum;
            }
            return;

        case ParseState::WitReadChecksum:
        {
            const uint8_t expected = static_cast<uint8_t>(running_sum_ & 0xFFU);
            if (byte == expected)
            {
                DecodeWitFrame(type_byte_, payload_buf_);
            }
            ResetParser();
            return;
        }
    }
}


void IMUDriver::DecodeWitFrame(uint8_t type, const uint8_t payload[8])
{
    switch (type)
    {
        case kWitTypeAccel:
        {
            const float ax = MakeInt16LE(payload[0], payload[1]) / 32768.0f *
                             kAccelScale_g * kGravity_mps2;
            const float ay = MakeInt16LE(payload[2], payload[3]) / 32768.0f *
                             kAccelScale_g * kGravity_mps2;
            const float az = MakeInt16LE(payload[4], payload[5]) / 32768.0f *
                             kAccelScale_g * kGravity_mps2;
            const std::array<float, 3> accel{ax, ay, az};
            if (!VectorWithinAbsLimit(accel, kMaxSafeAccelMps2))
            {
                RecordProtocolError("WIT accel outside safe range");
                return;
            }
            const uint64_t now_us = NowMonoUs();
            std::lock_guard<std::mutex> lock(sample_mutex_);
            latest_accel_ = accel;
            accel_update_us_ = now_us;
            accel_valid_ = true;
            break;
        }
        case kWitTypeGyro:
        {
            const float gx = MakeInt16LE(payload[0], payload[1]) / 32768.0f *
                             kGyroScale_dps * kDegToRad;
            const float gy = MakeInt16LE(payload[2], payload[3]) / 32768.0f *
                             kGyroScale_dps * kDegToRad;
            const float gz = MakeInt16LE(payload[4], payload[5]) / 32768.0f *
                             kGyroScale_dps * kDegToRad;
            const std::array<float, 3> gyro{gx, gy, gz};
            if (!VectorWithinAbsLimit(gyro, kMaxSafeGyroRadps))
            {
                RecordProtocolError("WIT gyro outside safe range");
                return;
            }
            const uint64_t now_us = NowMonoUs();
            std::lock_guard<std::mutex> lock(sample_mutex_);
            latest_gyro_ = gyro;
            gyro_update_us_ = now_us;
            gyro_valid_ = true;
            break;
        }
        case kWitTypeAngle:
        {
            const float roll = MakeInt16LE(payload[0], payload[1]) / 32768.0f *
                               kAngleScale_deg * kDegToRad;
            const float pitch = MakeInt16LE(payload[2], payload[3]) / 32768.0f *
                                kAngleScale_deg * kDegToRad;
            const float yaw = MakeInt16LE(payload[4], payload[5]) / 32768.0f *
                              kAngleScale_deg * kDegToRad;
            StoreEulerRadians(roll, pitch, yaw);
            break;
        }
        case kWitTypeQuat:
        {
            std::array<float, 4> quat{
                MakeInt16LE(payload[0], payload[1]) / 32768.0f,
                MakeInt16LE(payload[2], payload[3]) / 32768.0f,
                MakeInt16LE(payload[4], payload[5]) / 32768.0f,
                MakeInt16LE(payload[6], payload[7]) / 32768.0f};
            StoreQuaternion(quat);
            break;
        }
        case kWitTypeMag:
        default:
            break;
    }
}


void IMUDriver::DecodeFdiFrame(uint8_t type, const uint8_t *payload, std::size_t len)
{
    if (payload == nullptr)
    {
        return;
    }

    if ((type == kFdiTypeImu) && (len == kFdiImuPayloadLen))
    {
        const float gx = MakeFloatLE(&payload[kFdiImuGyroOffset + 0U]);
        const float gy = MakeFloatLE(&payload[kFdiImuGyroOffset + 4U]);
        const float gz = MakeFloatLE(&payload[kFdiImuGyroOffset + 8U]);
        const float ax = MakeFloatLE(&payload[kFdiImuAccelOffset + 0U]);
        const float ay = MakeFloatLE(&payload[kFdiImuAccelOffset + 4U]);
        const float az = MakeFloatLE(&payload[kFdiImuAccelOffset + 8U]);

        const std::array<float, 3> gyro{gx, gy, gz};
        const std::array<float, 3> accel{ax, ay, az};

        if (!VectorWithinAbsLimit(gyro, kMaxSafeGyroRadps))
        {
            RecordProtocolError("FDILink gyro outside safe range");
            return;
        }
        if (!VectorWithinAbsLimit(accel, kMaxSafeAccelMps2))
        {
            RecordProtocolError("FDILink accel outside safe range");
            return;
        }

        const uint64_t now_us = NowMonoUs();
        std::lock_guard<std::mutex> lock(sample_mutex_);
        latest_gyro_ = gyro;
        latest_accel_ = accel;
        gyro_update_us_ = now_us;
        accel_update_us_ = now_us;
        gyro_valid_ = true;
        accel_valid_ = true;
        return;
    }

    if ((type == kFdiTypeAhrs) && (len == kFdiAhrsPayloadLen))
    {
        std::array<float, 4> quat{
            MakeFloatLE(&payload[kFdiAhrsQuatOffset + 0U]),
            MakeFloatLE(&payload[kFdiAhrsQuatOffset + 4U]),
            MakeFloatLE(&payload[kFdiAhrsQuatOffset + 8U]),
            MakeFloatLE(&payload[kFdiAhrsQuatOffset + 12U])};

        if (NormalizeQuaternion(&quat))
        {
            StoreQuaternion(quat);
        }
        else
        {
            const float roll = MakeFloatLE(&payload[kFdiAhrsEulerOffset + 0U]);
            const float pitch = MakeFloatLE(&payload[kFdiAhrsEulerOffset + 4U]);
            const float yaw = MakeFloatLE(&payload[kFdiAhrsEulerOffset + 8U]);
            StoreEulerRadians(roll, pitch, yaw);
        }
    }
}


void IMUDriver::StoreQuaternion(std::array<float, 4> quat_wxyz)
{
    if (!NormalizeQuaternion(&quat_wxyz))
    {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(sample_mutex_);
        latest_quat_ = quat_wxyz;
        orientation_update_us_ = NowMonoUs();
        orientation_valid_ = true;
    }
}


void IMUDriver::StoreEulerRadians(float roll, float pitch, float yaw)
{
    if (!std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw))
    {
        return;
    }

    const float half_roll = 0.5f * roll;
    const float half_pitch = 0.5f * pitch;
    const float half_yaw = 0.5f * yaw;

    const float cr = std::cos(half_roll);
    const float sr = std::sin(half_roll);
    const float cp = std::cos(half_pitch);
    const float sp = std::sin(half_pitch);
    const float cy = std::cos(half_yaw);
    const float sy = std::sin(half_yaw);

    std::array<float, 4> quat{
        (cr * cp * cy) + (sr * sp * sy),
        (sr * cp * cy) - (cr * sp * sy),
        (cr * sp * cy) + (sr * cp * sy),
        (cr * cp * sy) - (sr * sp * cy)};
    StoreQuaternion(quat);
}