/**
 * @file proof.cpp
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-26
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#include "equihash_gpu/equihash/proof.h"

namespace Equihash
{
    Proof::Proof():solution_nonce_(0)
    {

    }

    Proof::Proof(const std::vector<uint8_t> & solution, uint32_t solution_nonce): solution_(solution), solution_nonce_(solution_nonce)
    {

    }

    Proof::~Proof()
    {

    }

    std::vector<uint8_t> & Proof::get_solution()
    {
        return solution_;
    }

    void Proof::set_solution(const std::vector<uint8_t> & sol)
    {
        solution_ = sol;
    }

    uint32_t Proof::get_solution_nonce()const
    {
        return solution_nonce_;
    }

    void Proof::set_solution_nonce(uint32_t nonce)
    {
        solution_nonce_ = nonce;
    }
}