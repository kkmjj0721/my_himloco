#include <linux/joystick.h>

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

namespace {

volatile sig_atomic_t g_running = 1;

void handle_signal(int /*signal_number*/)
{
    /* 信号处理函数中只修改 sig_atomic_t，避免不可重入操作。 */
    g_running = 0;
}

bool install_signal_handlers()
{
    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = handle_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;  // 不设置 SA_RESTART，Ctrl+C 可打断阻塞 read。

    if (sigaction(SIGINT, &action, nullptr) != 0) {
        return false;
    }

    if (sigaction(SIGTERM, &action, nullptr) != 0) {
        return false;
    }

    return true;
}

double normalize_axis_value(int16_t value)
{
    /*
     * Linux joystick 轴值为有符号 16 bit。
     * 负半轴包含 -32768，正半轴最大 32767，分母分开可保持端点落在 [-1, 1]。
     */
    if (value < 0) {
        return static_cast<double>(value) / 32768.0;
    }

    return static_cast<double>(value) / 32767.0;
}

void print_usage_hint(const char *program_name)
{
    std::cout << "用法: " << program_name << " [joystick设备路径]\n"
              << "默认设备: /dev/input/js0\n"
              << "示例: " << program_name << " /dev/input/js1\n\n";
}

void print_xbox_mapping_hint()
{
    std::cout << "常见 Xbox 蓝牙手柄映射参考（不同内核、蓝牙栈、模式可能不完全一致）:\n"
              << "  Axis 0: 左摇杆 X\n"
              << "  Axis 1: 左摇杆 Y\n"
              << "  Axis 2: LT 或触发器轴\n"
              << "  Axis 3: 右摇杆 X\n"
              << "  Axis 4: 右摇杆 Y\n"
              << "  Axis 5: RT 或触发器轴\n"
              << "  Axis 6/7: 十字键 X/Y（部分系统映射为按钮）\n"
              << "  Button 0/1/2/3: A/B/X/Y（常见映射）\n"
              << "  Button 4/5: LB/RB\n"
              << "  Button 6/7: View/Menu 或 Back/Start\n"
              << "  Button 8/9/10: Xbox/左摇杆按下/右摇杆按下（视驱动而定）\n\n";
}

void print_open_error(const std::string &device_path)
{
    std::cerr << "无法打开 joystick 设备: " << device_path << "\n"
              << "系统错误: " << std::strerror(errno) << "\n"
              << "请检查:\n"
              << "  1. Xbox 蓝牙手柄是否已连接并配对成功\n"
              << "  2. 设备节点是否存在: ls /dev/input/js*\n"
              << "  3. 当前用户是否有读取权限，可尝试加入 input 组或临时使用 sudo\n"
              << "  4. 可用 jstest 验证设备: jstest " << device_path << "\n";
}

void print_state_size(unsigned int axis_count, unsigned int button_count)
{
    std::cout << "状态数组: axes=" << axis_count
              << ", buttons=" << button_count << "\n";
}

}  // namespace

int main(int argc, char **argv)
{
    const std::string device_path = (argc > 1) ? argv[1] : "/dev/input/js0";

    if (argc > 2) {
        print_usage_hint(argv[0]);
        return 2;
    }

    if (!install_signal_handlers()) {
        std::cerr << "安装信号处理失败: " << std::strerror(errno) << "\n";
        return 1;
    }

    const int fd = open(device_path.c_str(), O_RDONLY);
    if (fd < 0) {
        print_open_error(device_path);
        return 1;
    }

    uint8_t axis_count = 0U;
    uint8_t button_count = 0U;
    char device_name[128];
    std::memset(device_name, 0, sizeof(device_name));

    if (ioctl(fd, JSIOCGAXES, &axis_count) != 0) {
        std::cerr << "警告: 查询轴数量失败: " << std::strerror(errno) << "\n";
    }

    if (ioctl(fd, JSIOCGBUTTONS, &button_count) != 0) {
        std::cerr << "警告: 查询按键数量失败: " << std::strerror(errno) << "\n";
    }

    if (ioctl(fd, JSIOCGNAME(sizeof(device_name)), device_name) != 0) {
        std::strncpy(device_name, "Unknown joystick", sizeof(device_name) - 1U);
        std::cerr << "警告: 查询设备名称失败: " << std::strerror(errno) << "\n";
    }

    std::vector<int16_t> axes(axis_count, 0);
    std::vector<uint8_t> buttons(button_count, 0U);

    std::cout << "Joystick 设备: " << device_path << "\n"
              << "设备名称: " << device_name << "\n"
              << "轴数量: " << static_cast<unsigned int>(axis_count) << "\n"
              << "按键数量: " << static_cast<unsigned int>(button_count) << "\n\n";

    print_xbox_mapping_hint();
    print_state_size(static_cast<unsigned int>(axes.size()),
                     static_cast<unsigned int>(buttons.size()));
    std::cout << "开始读取事件，按 Ctrl+C 退出。\n\n";

    while (g_running != 0) {
        js_event event;
        const ssize_t bytes = read(fd, &event, sizeof(event));

        if (bytes == static_cast<ssize_t>(sizeof(event))) {
            const bool is_initial_event = (event.type & JS_EVENT_INIT) != 0U;
            const uint8_t event_type = static_cast<uint8_t>(event.type & ~JS_EVENT_INIT);
            const unsigned int number = static_cast<unsigned int>(event.number);

            if (event_type == JS_EVENT_AXIS) {
                if (number >= axes.size()) {
                    axes.resize(number + 1U, 0);
                    print_state_size(static_cast<unsigned int>(axes.size()),
                                     static_cast<unsigned int>(buttons.size()));
                }

                axes[number] = event.value;
                const double normalized = normalize_axis_value(event.value);

                std::cout << (is_initial_event ? "[INIT] " : "")
                          << "AXIS  "
                          << "index=" << std::setw(3) << number
                          << " raw=" << std::setw(7) << event.value
                          << " norm=" << std::fixed << std::setprecision(3)
                          << std::setw(7) << normalized << "\n";
            } else if (event_type == JS_EVENT_BUTTON) {
                if (number >= buttons.size()) {
                    buttons.resize(number + 1U, 0U);
                    print_state_size(static_cast<unsigned int>(axes.size()),
                                     static_cast<unsigned int>(buttons.size()));
                }

                buttons[number] = (event.value != 0) ? 1U : 0U;

                std::cout << (is_initial_event ? "[INIT] " : "")
                          << "BUTTON"
                          << " index=" << std::setw(3) << number
                          << " value=" << static_cast<unsigned int>(buttons[number])
                          << "\n";
            } else {
                std::cout << (is_initial_event ? "[INIT] " : "")
                          << "UNKNOWN"
                          << " type=0x" << std::hex << static_cast<unsigned int>(event_type)
                          << std::dec
                          << " number=" << number
                          << " value=" << event.value << "\n";
            }
        } else if (bytes < 0) {
            if (errno == EINTR) {
                continue;
            }

            std::cerr << "读取 joystick 事件失败: " << std::strerror(errno) << "\n"
                      << "请检查蓝牙连接是否断开、设备节点是否仍存在，并可尝试:\n"
                      << "  ls /dev/input/js*\n"
                      << "  jstest " << device_path << "\n";
            close(fd);
            return 1;
        } else if (bytes == 0) {
            std::cerr << "joystick 设备结束读取，可能已断开连接。\n"
                      << "请检查蓝牙连接和 /dev/input/js* 设备节点。\n";
            close(fd);
            return 1;
        } else {
            std::cerr << "读取到不完整 joystick 事件: " << bytes
                      << " / " << sizeof(event) << " bytes\n";
        }
    }

    std::cout << "\n收到退出信号，关闭 joystick 设备。\n";
    close(fd);
    return 0;
}
