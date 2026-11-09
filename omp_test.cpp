#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num()
                      << " of " << omp_get_num_threads()
                      << '\n';
        }
    }

    return 0;
}