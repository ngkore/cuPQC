// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUHASH_HPP
#define CUHASH_HPP

#include "cupqc.hpp"

namespace cupqc {
    using SHA2_224    = decltype(Algorithm<algorithm::SHA2_32>() + SecurityCategory<1>() + Function<function::Hash>());
    using SHA2_256    = decltype(Algorithm<algorithm::SHA2_32>() + SecurityCategory<2>() + Function<function::Hash>());
    
    using SHA2_512_224  = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<1>() + Function<function::Hash>());
    using SHA2_512_256  = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<2>() + Function<function::Hash>());
    using SHA2_384    = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<4>() + Function<function::Hash>());
    using SHA2_512    = decltype(Algorithm<algorithm::SHA2_64>() + SecurityCategory<5>() + Function<function::Hash>());

    using SHA3_224    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<1>() + Function<function::Hash>());
    using SHA3_256    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<2>() + Function<function::Hash>());
    using SHA3_384    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<4>() + Function<function::Hash>());
    using SHA3_512    = decltype(Algorithm<algorithm::SHA3>()   + SecurityCategory<5>() + Function<function::Hash>());
    using SHAKE_128   = decltype(Algorithm<algorithm::SHAKE>()  + SecurityCategory<1>() + Function<function::Hash>());
    using SHAKE_256   = decltype(Algorithm<algorithm::SHAKE>()  + SecurityCategory<2>() + Function<function::Hash>());
}

#endif // CUHASH_HPP
