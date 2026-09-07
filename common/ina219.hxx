#ifndef INA219_HXX
#define INA219_HXX

#include <atomic>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

struct INA219Reading {
    double bus_voltage_v;
    double shunt_voltage_mv;
    double current_ma;
    double power_mw;
    // Wall-clock timestamp (microseconds since epoch) recorded at sample time.
    // Used to compute actual inter-sample dt for energy integration.
    int64_t timestamp_us;
};

struct INA219Stats {
    double bus_voltage_v_avg;
    double bus_voltage_v_min;
    double bus_voltage_v_max;
    double shunt_voltage_mv_avg;
    double current_ma_avg;
    double current_ma_min;
    double current_ma_max;
    double power_mw_avg;
    double power_mw_min;
    double power_mw_max;
    double energy_mj;      // average power over the window x window length
    double window_ms;      // length of the measured window
    int sample_count;      // readings taken inside the window
};

class INA219 {
   public:
    static constexpr uint8_t INA219_ADDR = 0x40;
    static constexpr uint8_t REG_CONFIG = 0x00;
    static constexpr uint8_t REG_SHUNT_V = 0x01;
    static constexpr uint8_t REG_BUS_V = 0x02;
    static constexpr uint8_t REG_POWER = 0x03;
    static constexpr uint8_t REG_CURRENT = 0x04;
    static constexpr uint8_t REG_CALIBRATION = 0x05;
    static constexpr uint16_t CAL_VALUE = 4096;
    static constexpr double CURRENT_LSB_MA = 0.1;
    static constexpr double POWER_LSB_MW = CURRENT_LSB_MA * 20.0;

    INA219() : i2c_fd_(-1) {}

    ~INA219() { close_device(); }

    bool open_device(const char* dev = "/dev/i2c-1") {
#ifdef __linux__
        i2c_fd_ = ::open(dev, O_RDWR);
        if (i2c_fd_ < 0) {
            return false;
        }
        if (ioctl(i2c_fd_, I2C_SLAVE, INA219_ADDR) < 0) {
            close_device();
            return false;
        }
        return true;
#else
        (void) dev;
        return false;
#endif
    }

    bool configure() {
#ifdef __linux__
        if (i2c_fd_ < 0) {
            return false;
        }
        if (!write_reg16(REG_CONFIG, 0x8000)) {
            return false;
        }
        usleep(10000);
        if (!write_reg16(REG_CONFIG, 0x399F)) {
            return false;
        }
        return write_reg16(REG_CALIBRATION, CAL_VALUE);
#else
        return false;
#endif
    }

    bool read_reading(INA219Reading& reading) {
#ifdef __linux__
        int16_t raw_bus, raw_shunt, raw_current;
        uint16_t raw_power;
        if (!read_reg16(REG_BUS_V, raw_bus)) {
            return false;
        }
        if (!read_reg16(REG_SHUNT_V, raw_shunt)) {
            return false;
        }
        if (!read_reg16(REG_CURRENT, raw_current)) {
            return false;
        }
        int16_t raw_power_signed;
        if (!read_reg16(REG_POWER, raw_power_signed)) {
            return false;
        }
        raw_power = (uint16_t) raw_power_signed;  // the power register is unsigned

        reading.bus_voltage_v = ((raw_bus >> 3) * 4) / 1000.0;
        reading.shunt_voltage_mv = raw_shunt * 0.01;
        reading.current_ma = raw_current * CURRENT_LSB_MA;
        reading.power_mw = raw_power * POWER_LSB_MW;
        return true;
#else
        (void) reading;
        return false;
#endif
    }

    void close_device() {
#ifdef __linux__
        if (i2c_fd_ >= 0) {
            ::close(i2c_fd_);
            i2c_fd_ = -1;
        }
#endif
    }

    bool is_open() const { return i2c_fd_ >= 0; }

   private:
    int i2c_fd_;

#ifdef __linux__
    bool write_reg16(uint8_t reg, uint16_t value) {
        uint8_t buf[3];
        buf[0] = reg;
        buf[1] = (value >> 8) & 0xFF;
        buf[2] = value & 0xFF;
        return ::write(i2c_fd_, buf, 3) == 3;
    }

    bool read_reg16(uint8_t reg, int16_t& value) {
        if (::write(i2c_fd_, &reg, 1) != 1) {
            return false;
        }
        uint8_t buf[2];
        if (::read(i2c_fd_, buf, 2) != 2) {
            return false;
        }
        value = (int16_t) ((buf[0] << 8) | buf[1]);
        return true;
    }
#endif
};

// Samples the sensor on a background thread. Meant for short windows: start()
// before the work, stop() after it, then get_stats(window_start, window_end)
// with the timestamps of the work (INA219Sampler::now_us()). Energy is the
// average power of the readings taken inside the window times the window
// length, so a window shorter than one sample interval still gets a sensible
// number; give it a window of at least ~100 ms to average over a few dozen
// readings (the pi's I2C bus at 100 kHz allows a reading every ~2-3 ms).
class INA219Sampler {
   public:
    typedef std::function<bool(INA219Reading&)> Reader;

    INA219Sampler() : running_(false), sample_interval_us_(1000) {}

    ~INA219Sampler() { stop(); }

    void set_sample_interval_us(int interval_us) { sample_interval_us_ = interval_us; }

    static int64_t now_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
    }

    bool start(INA219* sensor) {
        if (sensor == nullptr || !sensor->is_open()) {
            return false;
        }
        return start([sensor](INA219Reading& r) { return sensor->read_reading(r); });
    }

    // any reading source, e.g. a simulated sensor for testing the maths
    bool start(Reader reader) {
        if (running_) {
            return false;
        }
        reader_ = reader;
        {
            std::lock_guard<std::mutex> lock(readings_mutex_);
            readings_.clear();
        }
        running_ = true;
        thread_ = std::thread(&INA219Sampler::sample_loop, this);
        return true;
    }

    void stop() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    int total_readings() const {
        std::lock_guard<std::mutex> lock(readings_mutex_);
        return (int) readings_.size();
    }

    // statistics over the readings whose timestamp falls inside [window_start_us, window_end_us];
    // if none does (window shorter than a sample interval) the readings nearest the window are used.
    INA219Stats get_stats(int64_t window_start_us, int64_t window_end_us) const {
        std::lock_guard<std::mutex> lock(readings_mutex_);
        INA219Stats stats = {};
        stats.window_ms = (window_end_us - window_start_us) / 1000.0;
        if (readings_.empty()) {
            return stats;
        }

        std::vector<const INA219Reading*> in_window;
        for (const INA219Reading& r : readings_) {
            if (r.timestamp_us >= window_start_us && r.timestamp_us <= window_end_us) {
                in_window.push_back(&r);
            }
        }
        if (in_window.empty()) {
            // nearest reading before and after the window
            const INA219Reading* before = nullptr;
            const INA219Reading* after = nullptr;
            for (const INA219Reading& r : readings_) {
                if (r.timestamp_us < window_start_us) before = &r;
                if (r.timestamp_us > window_end_us && after == nullptr) after = &r;
            }
            if (before != nullptr) in_window.push_back(before);
            if (after != nullptr) in_window.push_back(after);
        }

        stats.sample_count = (int) in_window.size();
        stats.bus_voltage_v_min = stats.bus_voltage_v_max = in_window[0]->bus_voltage_v;
        stats.current_ma_min = stats.current_ma_max = in_window[0]->current_ma;
        stats.power_mw_min = stats.power_mw_max = in_window[0]->power_mw;
        double bus_v_sum = 0.0, shunt_mv_sum = 0.0, current_ma_sum = 0.0, power_mw_sum = 0.0;
        for (const INA219Reading* r : in_window) {
            bus_v_sum += r->bus_voltage_v;
            shunt_mv_sum += r->shunt_voltage_mv;
            current_ma_sum += r->current_ma;
            power_mw_sum += r->power_mw;
            if (r->bus_voltage_v < stats.bus_voltage_v_min) stats.bus_voltage_v_min = r->bus_voltage_v;
            if (r->bus_voltage_v > stats.bus_voltage_v_max) stats.bus_voltage_v_max = r->bus_voltage_v;
            if (r->current_ma < stats.current_ma_min) stats.current_ma_min = r->current_ma;
            if (r->current_ma > stats.current_ma_max) stats.current_ma_max = r->current_ma;
            if (r->power_mw < stats.power_mw_min) stats.power_mw_min = r->power_mw;
            if (r->power_mw > stats.power_mw_max) stats.power_mw_max = r->power_mw;
        }
        int n = stats.sample_count;
        stats.bus_voltage_v_avg = bus_v_sum / n;
        stats.shunt_voltage_mv_avg = shunt_mv_sum / n;
        stats.current_ma_avg = current_ma_sum / n;
        stats.power_mw_avg = power_mw_sum / n;
        stats.energy_mj = stats.power_mw_avg * (stats.window_ms / 1000.0);  // mW x s = mJ
        return stats;
    }

    // statistics over everything sampled since start()
    INA219Stats get_stats() const {
        int64_t first, last;
        {
            std::lock_guard<std::mutex> lock(readings_mutex_);
            if (readings_.empty()) {
                return INA219Stats{};
            }
            first = readings_.front().timestamp_us;
            last = readings_.back().timestamp_us;
        }
        return get_stats(first, last);
    }

   private:
    Reader reader_;
    std::atomic<bool> running_;
    int sample_interval_us_;
    std::thread thread_;
    mutable std::mutex readings_mutex_;
    std::vector<INA219Reading> readings_;

    void sample_loop() {
        while (running_) {
            INA219Reading reading;
            if (reader_(reading)) {
                reading.timestamp_us = now_us();
                std::lock_guard<std::mutex> lock(readings_mutex_);
                readings_.push_back(reading);
            }
            // sleep in short slices so stop() returns promptly
            int slept = 0;
            while (running_ && slept < sample_interval_us_) {
                int slice = std::min(1000, sample_interval_us_ - slept);
                std::this_thread::sleep_for(std::chrono::microseconds(slice));
                slept += slice;
            }
        }
    }
};

#endif
