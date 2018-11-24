/**
 * @file equihash_solver.h
 * @author ofir iluz (iluzofir@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2018-11-24
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#ifndef EQUIHASHGPU_EQUIHASH_SOLVER_H_
#define EQUIHASHGPU_EQUIHASH_SOLVER_H_

#include "equihash_gpu/equihash/proof.h"

namespace Equihash
{
    class IEquihashSolver
    {
    public:
        IEquihashSolver();
        virtual ~IEquihashSolver();

        virtual Proof find_proof() = 0;
        virtual bool verify_proof(const Proof & proof) = 0;
    };
}

#endif