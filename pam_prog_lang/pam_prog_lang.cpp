#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

#include <immintrin.h>

#ifdef _WIN32
#include <Windows.h>
// Windows.h includes a lot of annoying preprocessor defines
#undef min
#undef max
#else
#include <fstream>
#endif

using namespace std;

class KMedoids {
public:
    KMedoids(string filename, size_t size) : size(size), distances(new float[size * size]), k(0), closest(nullptr), closest_dist(nullptr), second_closest(nullptr)
    {
#ifdef _WIN32
        // Use Windows' file mapping because it's MUCH faster than using C++'s file IO streams: 6s vs 32s
        HANDLE file = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (file == NULL)
            exit(-1);
        LARGE_INTEGER file_size;
        GetFileSizeEx(file, &file_size);
        HANDLE map = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
        if (map == NULL)
            exit(-1);
        char* data = static_cast<char*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
        if (data == NULL)
            exit(-1);
        char* end, * orig = data;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < i; j++) {
                distances[i * size + j] = strtof(data, &end);
                distances[j * size + i] = distances[i * size + j];
                while (end - orig < file_size.QuadPart && (*end == ',' || isspace(static_cast<unsigned char>(*end))))
                    end++;
                data = end;
            }
            distances[i * size + i] = 0;
            while (*data != '\n')
                data++;
        }

        UnmapViewOfFile(orig);
        CloseHandle(map);
        CloseHandle(file);
#else
        // Slower fallback implementation
        ifstream file(filename, ios_base::in);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < i; j++) {
                file >> distances[i * size + j];
                // Distances matrix is symmetric
                distances[j * size + i] = distances[i * size + j];
                while (!file.eof()) {
                    int peek = static_cast<unsigned char>(file.peek());
                    if (peek != ',' && !isspace(peek))
                        break;
                    file.get();
                }
            }
            distances[i * size + i] = 0;
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
#endif
    }

    ~KMedoids()
    {
        if (distances) {
            delete[] distances;
            distances = nullptr;
        }
    }

    int optimize(double& silhouette)
    {
        double max_s = -INFINITY;
        int max_k = 0;

        closest = new int[size];
        closest_dist = new float[size];
        second_closest = new float[size];

        for (int k = 2, count = 5; count > 0; k++, count--) {
            cout << "Trying k=" << k << "...\t";
            build(k);
            swap();
            double curr = silhouette_coef();
            cout << "Silhouette coef: " << curr << endl;
            if (curr > max_s) {
                max_s = curr;
                max_k = k;
                count = 5;
            }
        }
        delete[] closest;
        delete[] closest_dist;
        delete[] second_closest;
        silhouette = max_s;
        return max_k;
    }

private:
    void update_closests()
    {
        __m256 inf = _mm256_set1_ps(INFINITY);
        for (size_t i = 0; i + 7 < size; i += 8) {
            _mm256_storeu_ps(&closest_dist[i], inf);
            _mm256_storeu_ps(&second_closest[i], inf);
        }

        for (size_t i = size - size % 8; i < size; i++)
            closest_dist[i] = second_closest[i] = INFINITY;

        for (size_t i = 0; i < size; i++) {
            for (int j = 0; j < medoids.size(); j++) {
                int medoid = medoids[j];
                if (distances[i * size + medoid] < closest_dist[i]) {
                    second_closest[i] = closest_dist[i];
                    closest_dist[i] = distances[i * size + medoid];
                    closest[i] = j;
                }
                else if (distances[i * size + medoid] < second_closest[i]) {
                    second_closest[i] = distances[i * size + medoid];
                }
            }
        }
    }

    void add_medoid(int point)
    {
        medoids.push_back(point);
        update_closests();
    }

    void swap_medoid(size_t i, int h)
    {
        medoids[i] = h;
        update_closests();
    }

    float horizontal_sum_vector(__m256 v)
    {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        vlow = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(vlow);
        __m128 sums = _mm_add_ps(vlow, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    void build(int k)
    {
        medoids = vector<int>();
        this->k = k;

        float min_val = INFINITY;
        int smallest = 0;
        for (int i = 0; i < size; i++) {
            __m256 sum = _mm256_setzero_ps();
            for (size_t j = 0; j + 7 < size; j += 8) {
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(&distances[i * size + j]));
            }

            float hsum = horizontal_sum_vector(sum);
            for (size_t j = size - size % 8; j < size; j++)
                hsum += distances[i * size + j];

            if (hsum < min_val) {
                min_val = hsum;
                smallest = i;
            }
        }

        add_medoid(smallest);

        for (int i = 0; i < k - 1; i++) {
            double tds = INFINITY;
            int candidate = 0;
            for (int c = 0; c < size; c++) {
                if (find(medoids.begin(), medoids.end(), c) != medoids.end())
                    continue;

                __m256 td = _mm256_setzero_ps();

                for (size_t j = 0; j + 7 < size; j += 8) {
                    __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(&distances[c * size + j]), _mm256_loadu_ps(&closest_dist[j]));
                    __m256 zero = _mm256_setzero_ps();
                    delta = _mm256_min_ps(delta, zero);

                    td = _mm256_add_ps(td, delta);
                }
                double htd = horizontal_sum_vector(td);
                // Cancel out the erroneous addition of -closest_dist[c] to td in the above loop
                htd += closest_dist[c];
                for (size_t j = size - size % 8; j < size; j++) {
                    float delta = distances[c * size + j] - closest_dist[j];
                    if (delta < 0)
                        htd += delta;
                }

                if (htd < tds) {
                    tds = htd;
                    candidate = c;
                }
            }
            add_medoid(candidate);
        }
    }

    void computeLoss(vector<double>& losses)
    {
        losses.assign(k, 0.0);
        for (int i = 0; i < size; i++) {
            losses[closest[i]] += static_cast<double>(second_closest[i]) - closest_dist[i];
        }
    }

    void swap()
    {
        int last = -1;
        bool first = true;
        vector<double> tds(k, 0.0);
        const __m256 zero = _mm256_setzero_ps();

        computeLoss(tds);

        while (true) {
            for (int i = 0; i < size; i++) {
                if (i == last || (last == -1 && !first))
                    return;
                if (find(medoids.begin(), medoids.end(), i) != medoids.end())
                    continue;


                vector<double> dtd = tds;
                __m256 vtdc = _mm256_setzero_ps();

                for (size_t j = 0; j + 7 < size; j += 8) {
                    __m256 nearest_dist = _mm256_loadu_ps(&closest_dist[j]);
                    __m256 second = _mm256_loadu_ps(&second_closest[j]);
                    __m256 dist = _mm256_loadu_ps(&distances[i * size + j]);
                    __m256 diff = _mm256_sub_ps(dist, nearest_dist);
                    __m256 mask1 = _mm256_cmp_ps(diff, zero, _CMP_LT_OQ);
                    vtdc = _mm256_add_ps(vtdc, _mm256_and_ps(diff, mask1));

                    __m256 ns_diff = _mm256_sub_ps(nearest_dist, second);
                    ns_diff = _mm256_and_ps(ns_diff, mask1);
                    __m256 sdist = _mm256_sub_ps(dist, second);
                    __m256 mask2 = _mm256_cmp_ps(sdist, zero, _CMP_LT_OQ);
                    sdist = _mm256_and_ps(mask2, _mm256_andnot_ps(mask1, sdist));

                    __m256 total = _mm256_add_ps(ns_diff, sdist);
                    float res[8];
                    _mm256_storeu_ps(res, total);
                    for (int m = 0; m < 8; m++)
                        dtd[closest[j + m]] += res[m];
                }

                double tdc = horizontal_sum_vector(vtdc);
                for (size_t j = size - size % 8; j < size; j++) {
                    double dist = distances[i * size + j];
                    if (dist < closest_dist[j]) {
                        tdc += dist - closest_dist[j];
                        dtd[closest[j]] += static_cast<double>(closest_dist[j]) - second_closest[j];
                    }
                    else if (dist < second_closest[j]) {
                        dtd[closest[j]] += dist - second_closest[j];
                    }
                }

                size_t argmin = min_element(dtd.begin(), dtd.end()) - dtd.begin();
                dtd[argmin] += tdc;
                if (dtd[argmin] < -0.00001) {
                    swap_medoid(argmin, i);
                    computeLoss(tds);
                    last = i;
                }
            }
            first = false;
        }
    }

    double silhouette_a(int i, size_t& clusterSize)
    {
        size_t cluster_size = 0;
        __m256 vsum = _mm256_setzero_ps();
        const __m256i cluster = _mm256_set1_epi32(closest[i]);

        for (size_t j = 0; j + 7 < size; j += 8) {
            __m256i jclust = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&closest[j]));
            __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(jclust, cluster));
            __m256 masked_dist = _mm256_and_ps(_mm256_loadu_ps(&distances[i * size + j]), mask);
            vsum = _mm256_add_ps(vsum, masked_dist);

            int bits = _mm256_movemask_ps(mask);
            cluster_size += _mm_popcnt_u32(bits);
        }
        double sum = horizontal_sum_vector(vsum);
        for (size_t j = size - size % 8; j < size; j++) {
            if (closest[j] != closest[i])
                continue;

            cluster_size++;
            sum += distances[i * size + j];
        }

        clusterSize = cluster_size;
        if (clusterSize == 1)
            return 0;
        return sum / (cluster_size - 1);
    }

    double silhouette_b(int i)
    {
        double min_val = INFINITY;
        for (int m = 0; m < medoids.size(); m++) {
            if (m == closest[i])
                continue;

            int count = 0;
            __m256 vsum = _mm256_setzero_ps();
            __m256i cluster = _mm256_set1_epi32(m);

            for (size_t j = 0; j + 7 < size; j += 8) {
                __m256i jclust = _mm256_loadu_si256(reinterpret_cast<__m256i*>(&closest[j]));
                __m256 mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(jclust, cluster));
                __m256 masked_dist = _mm256_and_ps(_mm256_loadu_ps(&distances[i * size + j]), mask);
                vsum = _mm256_add_ps(vsum, masked_dist);

                int bits = _mm256_movemask_ps(mask);
                count += _mm_popcnt_u32(bits);
            }
            double sum = horizontal_sum_vector(vsum);
            for (size_t j = size - size % 8; j < size; j++) {
                if (closest[j] != m)
                    continue;

                count++;
                sum += distances[i * size + j];
            }

            double mean = sum / count;
            if (mean < min_val)
                min_val = mean;
        }
        return min_val;
    }

    double silhouette_s(int i)
    {
        size_t cluster_size;
        double a = silhouette_a(i, cluster_size);
        if (cluster_size == 1)
            return 0;

        double b = silhouette_b(i);

        return (b - a) / max(a, b);
    }

    double silhouette_coef()
    {
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += silhouette_s(i);
        }
        return sum / size;
    }

    float* distances;
    size_t size;
    int k;
    vector<int> medoids;
    int* closest;
    float* closest_dist;
    float* second_closest;
};

int main()
{
    cout << "Loading data... ";
    KMedoids km("dist_matrix(10000x10000).txt", 10000);
    cout << "Done" << endl;
    cout << "Finding optimal k" << endl;
    double silhouette;
    int res = km.optimize(silhouette);
    cout << endl << "Final result: k=" << res << " with silhouette coefficient: " << silhouette << endl;
    return 0;
}
