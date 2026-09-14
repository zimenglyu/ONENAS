#ifndef ONENAS_PI_PROTOCOL_HXX
#define ONENAS_PI_PROTOCOL_HXX

#include <cstdint>
#include <cstring>

#include <string>
using std::string;

#include <vector>
using std::vector;

const int32_t PI_MAGIC = 0x4F4E5049;  // "ONPI"
const int32_t PI_VERSION = 2;
const int32_t PI_MODE_GLOBAL_BEST = 0;
const int32_t PI_MODE_ISLAND_BEST = 1;

inline string pi_mode_name(int32_t mode) {
    return mode == PI_MODE_ISLAND_BEST ? "island_best" : "global_best";
}

struct PiGenerationMessage {
    int32_t generation = -1;
    int32_t mode = PI_MODE_GLOBAL_BEST;
    int32_t seed = -1;
    int32_t num_episodes_total = 0;
    vector<int32_t> test_episode_ids;
    vector<int32_t> islands;                  // one per genome
    vector<vector<char> > genome_bytes;  // one per genome

    void add_genome(int32_t island, const char* bytes, int32_t length) {
        islands.push_back(island);
        genome_bytes.push_back(vector<char>(bytes, bytes + length));
    }

    vector<char> serialize() const {
        vector<char> out;
        auto put = [&out](int32_t v) {
            const char* p = reinterpret_cast<const char*>(&v);
            out.insert(out.end(), p, p + sizeof(int32_t));
        };
        put(PI_MAGIC);
        put(PI_VERSION);
        put(generation);
        put(mode);
        put(seed);
        put(num_episodes_total);
        put((int32_t) test_episode_ids.size());
        for (int32_t id : test_episode_ids) {
            put(id);
        }
        put((int32_t) genome_bytes.size());
        for (size_t g = 0; g < genome_bytes.size(); g++) {
            put(islands[g]);
            put((int32_t) genome_bytes[g].size());
            out.insert(out.end(), genome_bytes[g].begin(), genome_bytes[g].end());
        }
        return out;
    }

    // returns false (and sets error) on a malformed buffer
    bool parse(const char* data, int32_t length, string& error) {
        int32_t pos = 0;
        auto get = [&](int32_t& v) -> bool {
            if (pos + (int32_t) sizeof(int32_t) > length) {
                return false;
            }
            memcpy(&v, data + pos, sizeof(int32_t));
            pos += sizeof(int32_t);
            return true;
        };
        int32_t magic, version, n;
        if (!get(magic) || magic != PI_MAGIC) {
            error = "bad magic";
            return false;
        }
        if (!get(version) || version != PI_VERSION) {
            error = "unsupported version";
            return false;
        }
        if (!get(generation) || !get(mode) || !get(seed) || !get(num_episodes_total) || !get(n) || n < 0) {
            error = "bad header";
            return false;
        }
        test_episode_ids.assign(n, 0);
        for (int32_t i = 0; i < n; i++) {
            if (!get(test_episode_ids[i])) {
                error = "truncated test ids";
                return false;
            }
        }
        if (!get(n) || n < 0) {
            error = "bad genome count";
            return false;
        }
        islands.clear();
        genome_bytes.clear();
        for (int32_t g = 0; g < n; g++) {
            int32_t island, len;
            if (!get(island) || !get(len) || len < 0 || pos + len > length) {
                error = "truncated genome";
                return false;
            }
            islands.push_back(island);
            genome_bytes.push_back(vector<char>(data + pos, data + pos + len));
            pos += len;
        }
        return true;
    }
};

#endif
