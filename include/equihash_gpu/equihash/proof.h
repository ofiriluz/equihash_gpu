/**
 * @file proof.h
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#ifndef EQUIHASHGPU_PROOF_H_
#define EQUIHASHGPU_PROOF_H_

#include <vector>
#include <stdint.h>

namespace Equihash
{
    class Proof
    {
    private:
        std::vector<uint8_t> solution_;
        uint32_t solution_nonce_;

    public:
        Proof();
        Proof(const std::vector<uint8_t> & solutions, uint32_t solution_nonce);
        virtual ~Proof();

        std::vector<uint8_t> & get_solution();
        void set_solution(const std::vector<uint8_t> & sol);
        uint32_t get_solution_nonce()const;
        void set_solution_nonce(uint32_t nonce);
    };
}

#endif