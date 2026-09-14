#ifndef ONENAS_PI_SENDER_HXX
#define ONENAS_PI_SENDER_HXX

#include <netdb.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <condition_variable>
using std::condition_variable;

#include <cstring>

#include <deque>
using std::deque;

#include <mutex>
using std::lock_guard;
using std::mutex;
using std::unique_lock;

#include <string>
using std::string;
using std::to_string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"

class PiSender {
   private:
    string host;
    string port;
    int32_t sockfd = -1;
    bool stopping = false;
    mutex m;
    condition_variable cv;
    deque<vector<char> > queue;
    thread worker;

    bool connect_to_pi() {
        struct addrinfo hints = {}, *res = NULL;
        hints.ai_socktype = SOCK_STREAM;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
            return false;
        }
        for (struct addrinfo* r = res; r != NULL && sockfd < 0; r = r->ai_next) {
            int32_t fd = socket(r->ai_family, r->ai_socktype, r->ai_protocol);
            if (fd >= 0 && connect(fd, r->ai_addr, r->ai_addrlen) == 0) {
                sockfd = fd;
            } else if (fd >= 0) {
                close(fd);
            }
        }
        freeaddrinfo(res);
        if (sockfd < 0) {
            Log::warning("pi sender could not connect to %s:%s, will retry\n", host.c_str(), port.c_str());
        } else {
            Log::info("pi sender connected to %s:%s\n", host.c_str(), port.c_str());
        }
        return sockfd >= 0;
    }

    bool send_all(const vector<char>& msg) {
        size_t total = 0;
        while (total < msg.size()) {
            ssize_t n = send(sockfd, msg.data() + total, msg.size() - total, 0);
            if (n <= 0) {
                close(sockfd);
                sockfd = -1;
                return false;
            }
            total += n;
        }
        return true;
    }

    void run() {
        signal(SIGPIPE, SIG_IGN);  // a dropped pi connection must not kill the master
        Log::set_id("pi_sender");
        while (true) {
            vector<char> msg;
            {
                unique_lock<mutex> lock(m);
                cv.wait(lock, [this] { return stopping || !queue.empty(); });
                if (queue.empty()) {
                    Log::release_id("pi_sender");
                    return;
                }
                msg = queue.front();
            }
            if ((sockfd >= 0 || connect_to_pi()) && send_all(msg)) {
                lock_guard<mutex> lock(m);
                queue.pop_front();
            } else if (stopping) {
                Log::release_id("pi_sender");
                return;
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
    }

   public:
    PiSender(string _host, int32_t _port) : host(_host), port(to_string(_port)) {
        worker = thread(&PiSender::run, this);
    }

    ~PiSender() {
        {
            lock_guard<mutex> lock(m);
            stopping = true;
        }
        cv.notify_all();
        worker.join();
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    // copies the bytes and returns immediately
    void enqueue(const char* bytes, int32_t length) {
        vector<char> msg(sizeof(int32_t) + length);
        memcpy(msg.data(), &length, sizeof(int32_t));
        memcpy(msg.data() + sizeof(int32_t), bytes, length);
        {
            lock_guard<mutex> lock(m);
            queue.push_back(std::move(msg));
        }
        cv.notify_one();
    }
};

#endif
