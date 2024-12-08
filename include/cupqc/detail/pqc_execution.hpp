// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUPQC_DETAIL_PQC_EXECUTION_HPP
#define CUPQC_DETAIL_PQC_EXECUTION_HPP

#include "pqc_description.hpp"
#include "database.hpp"
#include <cstdint>

namespace cupqc {
    namespace detail {

        template<class... Operators>
        class pqc_execution: public pqc_full_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type = pqc_full_description<Operators...>;
            using this_type = pqc_execution<Operators...>;

        protected:

            /// ---- Constraints

            // We need Block operator to be specified exactly once
            static constexpr bool has_one_block = has_at_most_one_of<operator_type::block, this_type>::value;
            static_assert(has_one_block, "Can't create pqc function with two execution operators");
        };


        template<class... Operators>
        class pqc_block_execution: public pqc_execution<Operators...>
        {

            using this_type = pqc_block_execution<Operators...>;
            using base_type = pqc_execution<Operators...>;

            /// ---- Traits

            // Block Dimension
            // * Default value: selected by implementation
            static constexpr bool has_block_dim        = has_operator<operator_type::block_dim, base_type>::value;
            using default_pqc_block_dim                = BlockDim<128>;
            using this_pqc_block_dim                   = get_or_default_t<operator_type::block_dim, base_type, default_pqc_block_dim>;
            static constexpr auto this_pqc_block_dim_v = this_pqc_block_dim::value;

            // Batches per Block
            // * Default value: 1
            static constexpr bool has_batches_per_block        = has_operator<operator_type::batches_per_block, base_type>::value;
            using default_pqc_batches_per_block                = BatchesPerBlock<128>;
            using this_pqc_batches_per_block                   = get_or_default_t<operator_type::batches_per_block, base_type, default_pqc_batches_per_block>;
            static constexpr auto this_pqc_batches_per_block_v = this_pqc_batches_per_block::value;

            /// ---- Constraints
            static_assert(this_pqc_block_dim::y == 1 && this_pqc_block_dim::z == 1,
                          "Provided block dimension is invalid, y and z dimensions must both be 1.");
            static constexpr bool valid_block_dim = this_pqc_block_dim::flat_size >= 32 && this_pqc_block_dim::flat_size <= 1024;
            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't have more than 1024 threads.");

            static_assert(this_pqc_batches_per_block_v > 0, "Providing number of batches per block is invalid.  BatchesPerBlock<0> is unsupported.");

            template<class this_t, function func, class T = void>
            using function_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<this_t::is_complete && this_t::this_pqc_function_v == func, T>;

            /// ---- Accessors
        public:
            static constexpr auto BlockDim = this_pqc_block_dim_v;
            static constexpr auto SecurityCategory = base_type::this_pqc_security_category_v;

            /// ---- Execution
        public:
            static constexpr size_t workspace_size = database::global_memory_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();
            static constexpr size_t shared_memory_size = database::shared_memory_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();
            static constexpr size_t entropy_size = database::entropy_size<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_function_v>();

            // keygen
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* public_key, uint8_t* secret_key,
                                           uint8_t* entropy,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Keygen> {
                database::keygen<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(public_key, secret_key, entropy, workspace, smem_workspace);
            }

            // encaps
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* cipher_text, uint8_t* shared_secret, const uint8_t* public_key,
                                           uint8_t* entropy,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Encaps> {
                database::encaps<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(cipher_text, shared_secret, public_key, entropy, workspace, smem_workspace);
            }

            // decaps
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* shared_secret, const uint8_t* ciphertext, const uint8_t* secret_key,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Decaps> {
                database::decaps<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(shared_secret, ciphertext, secret_key, workspace, smem_workspace);
            }

            // sign
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(uint8_t* signature, const uint8_t* message, const size_t message_length, const uint8_t* secret_key,
                                           uint8_t* entropy, uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Sign> {
                database::sign<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(signature, message, message_length, secret_key, entropy, workspace, smem_workspace);
            }

            // verify
            // N.B., have to use a template to use SFINAE
            template<class this_t = this_type>
            inline __device__ auto execute(const uint8_t* message, const size_t message_length, const uint8_t* signature, const uint8_t* public_key,
                                           uint8_t* workspace, uint8_t* smem_workspace)
                    -> function_enable_if_t<this_t, function::Verify, bool> {
                return database::verify<this_type::this_pqc_algorithm_v, this_type::this_pqc_security_category_v, this_type::this_pqc_block_dim_v.x>(message, message_length, signature, public_key, workspace, smem_workspace);
            }
        };


        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator     = has_operator<operator_type::block, pqc_operator_wrapper<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator;

            // TODO cuBLASDx conditionally instantiates this, check if cuPQCDx also needs conditional instantiation
            using execution_type = pqc_block_execution<Operators...>;
            using description_type = pqc_full_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;

    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::pqc_full_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::pqc_full_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::pqc_full_description<Operators1...>&,
                                                       const detail::pqc_full_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cupqc

#endif // CUPQC_DETAIL_PQC_EXECUTION_HPP

