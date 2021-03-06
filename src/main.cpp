#include <equihash_gpu/equihash/gpu/equihash_gpu_solver.h>
#include <stdio.h>

int main(int argc, char ** argv)
{
    uint32_t n = 0, k=0;
    uint32_t seed[SEED_SIZE] = {0, 0, 0, 0};
    if (argc < 2) 
    {
        return 1;
    }

    /* parse options */
    for (int i = 1; i < argc; i++)
    {
        const char *a = argv[i];
        unsigned long input = 0;
        if (!strcmp(a, "-n"))
        {
            if (i < argc - 1) 
            {
                i++;
                input = strtoul(argv[i], NULL, 10);
                n = input;
                continue;
            }
            else 
            {
                printf("-n is mandatory");
                return 1;
            }
        }
        else if (!strcmp(a, "-k")) 
        {
            if (i < argc - 1) 
            {
                i++;
                input = strtoul(argv[i], NULL, 10);
                k = input;
                continue;
            }
            else {
                printf("-k is mandatory");
                return 1;
            }
        }
        if (!strcmp(a, "-s")) {
            if (i < argc - 1) {
                i++;
                input = strtoul(argv[i], NULL, 10);
                if (input == 0 ||
                    input > 0xFFFFFF) {
                    printf("bad numeric input for -s");
                    return 1;
                }
                for(size_t j=0;j<SEED_SIZE;j++)
                {
                    seed[j] = input;
                }
                continue;
            }
            else {
                printf("missing -s argument");
                return 1;
            }
        }
    }

    printf("N = %d\n", n);
    printf("K = %d\n", k);
    printf("Seed = ");
    for(size_t j=0;j<SEED_SIZE;j++)
    {
        printf("%d  ", seed[j]);
    }
    printf("\n");
    Equihash::EquihashGPUSolver solver(n, k, seed);
    solver.find_proof();
}