// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUPQC_DETAIL_PQC_DESCRIPTION_HPP
#define CUPQC_DETAIL_PQC_DESCRIPTION_HPP

#include "commondx/traits/detail/get.hpp"

#include "../operators.hpp"
#include "../traits/detail/description_traits.hpp"

#include <cstdint>

namespace cupqc {
    namespace detail {

        template<class... Operators>
        class pqc_operator_wrapper: public commondx::detail::description_expression { };

        template<class... Operators>
        class pqc_description: public commondx::detail::description_expression {

            using description_type = pqc_operator_wrapper<Operators...>;

            // Friend declarations
			template<class enable, class... Ops>
			friend class pqc_full_description_helper;
            template<class... Ops>
            friend constexpr bool has_algorithm(algorithm alg);
            template<class... Ops>
            friend constexpr bool has_no_algorithm();

        protected:

            // Algorithm
            // * Default value: NONE
            // * Dummy value: ML_KEM
            static constexpr bool has_algorithm        = has_operator<operator_type::algorithm, description_type>::value;
            using dummy_default_pqc_algorithm          = Algorithm<algorithm::ML_KEM>;
            using this_pqc_algorithm                   = get_or_default_t<operator_type::algorithm, description_type, dummy_default_pqc_algorithm>;
            static constexpr auto this_pqc_algorithm_v = this_pqc_algorithm::value;

            // Security Level
            // * Default value: NONE
            // * Dummy value: 128
            static constexpr bool has_security_category        = has_operator<operator_type::security_category, description_type>::value;
            using dummy_default_pqc_security_category          = SecurityCategory<3>;
            using this_pqc_security_category                   = get_or_default_t<operator_type::security_category, description_type, dummy_default_pqc_security_category>;
            static constexpr auto this_pqc_security_category_v = this_pqc_security_category::value;

            // Function
            // * Default value: NONE
            // * Dummy value: Keygen
            // static constexpr bool has_function        = has_operator<operator_type::function, description_type>::value || is_hash_algorithm(this_pqc_algorithm_v);
            // using dummy_default_pqc_function          = COMMONDX_STL_NAMESPACE::conditional_t<is_hash_algorithm(this_pqc_algorithm_v), Function<function::Hash>, Function<function::Keygen>>;
            static constexpr bool has_function        = has_operator<operator_type::function, description_type>::value;
            using dummy_default_pqc_function          = Function<function::Keygen>;
            using this_pqc_function                   = get_or_default_t<operator_type::function, description_type, dummy_default_pqc_function>;
            static constexpr auto this_pqc_function_v = this_pqc_function::value;

            // SM
            // * Default value: NONE
            // * Dummy value: 700
            static constexpr bool has_sm         = has_operator<operator_type::sm, description_type>::value;
            using dummy_default_pqc_sm          = SM<700>;
            using this_pqc_sm                   = get_or_default_t<operator_type::sm, description_type, dummy_default_pqc_sm>;
            static constexpr auto this_pqc_sm_v = this_pqc_sm::value;

            // True if description is complete description
            static constexpr bool is_complete = has_algorithm && has_security_category && has_function;

            /// ---- Constraints

            // We can only have one of each option

            static constexpr bool has_one_algorithm      = has_at_most_one_of<operator_type::algorithm,      description_type>::value;
            static constexpr bool has_one_security_category = has_at_most_one_of<operator_type::security_category, description_type>::value;
            static constexpr bool has_one_function       = has_at_most_one_of<operator_type::function,       description_type>::value;
            static constexpr bool has_one_sm             = has_at_most_one_of<operator_type::sm,             description_type>::value;

            static_assert(has_one_algorithm,      "Can't create pqc function with two Algorithm<> expressions");
            static_assert(has_one_security_category, "Can't create pqc function with two SecurityCategory<> expressions");
            static_assert(has_one_function,       "Can't create pqc function with two Function<> expressions");
            static_assert(has_one_sm,             "Can't create pqc function with two SM<> expressions");

            // Check that function is compatible with algorithm
            static_assert(this_pqc_function_v != function::Encaps || is_kem_algorithm(this_pqc_algorithm_v), "Can't create pqc function with Function<Encaps> and a non KEM algorithm");
            static_assert(this_pqc_function_v != function::Decaps || is_kem_algorithm(this_pqc_algorithm_v), "Can't create pqc function with Function<Decaps> and a non KEM algorithm");
            static_assert(this_pqc_function_v != function::Sign   || is_dss_algorithm(this_pqc_algorithm_v), "Can't create pqc function with Function<Sign> and a non DSS algorithm");
            static_assert(this_pqc_function_v != function::Verify || is_dss_algorithm(this_pqc_algorithm_v), "Can't create pqc function with Function<Verify> and a non DSS algorithm");
            static_assert(this_pqc_function_v != function::Hash   || is_hash_algorithm(this_pqc_algorithm_v), "Can't create pqc function with Function<Hash> and a non hashing algorithm");

            /// ---- End of Constraints

        };

        template<>
        class pqc_description<>: public commondx::detail::description_expression {};


        // Wrapper type to handle some of the algorithm specific parameters (e.g., sizes)
        // Making a helper type is needed in order to conditionally create variables

        template<class... Operators>
        inline constexpr bool has_algorithm(algorithm alg) {
            return pqc_description<Operators...>::has_algorithm
                   && pqc_description<Operators...>::this_pqc_algorithm_v == alg
                   && pqc_description<Operators...>::has_security_category;
        }
        template<class... Operators>
        inline constexpr bool has_no_algorithm() {
            return !pqc_description<Operators...>::has_algorithm
                   || !pqc_description<Operators...>::has_security_category;
        }

        template<class enable, class... Operators>
        class pqc_full_description_helper;

        template<class... Operators>
        using pqc_full_description = pqc_full_description_helper<void, Operators...>;

        // Both algorithm and security level are required for parameters
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_no_algorithm<Operators...>()>, Operators...>
                                   : public pqc_description<Operators...> {
        };

        // Kyber parameters
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::ML_KEM)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 1 || sec_category == 3 || sec_category == 5, "Kyber only supports security levels 1, 3, 5");

        public:
            // TODO can we avoid the duplication with kem_kyber_parameters.hpp, without leaking anything else in that file?

            static constexpr size_t public_key_size = (sec_category == 1) ?  800 : ((sec_category == 3) ? 1184 : 1568);
            static constexpr size_t secret_key_size = (sec_category == 1) ? 1632 : ((sec_category == 3) ? 2400 : 3168);
            static constexpr size_t ciphertext_size = (sec_category == 1) ?  768 : ((sec_category == 3) ? 1088 : 1568);
            static constexpr size_t shared_secret_size = 32;
        };

        // ML-DSA parameters
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::ML_DSA)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 2 || sec_category == 3 || sec_category == 5, "ML-DSA only supports security levels 2, 3, 5");

        public:
            static constexpr size_t public_key_size = (sec_category == 2) ? 1312 : ((sec_category == 3) ? 1952 : 2592);
            static constexpr size_t secret_key_size = (sec_category == 2) ? 2560 : ((sec_category == 3) ? 4032 : 4896);
            static constexpr size_t signature_size  = (sec_category == 2) ? 2420 : ((sec_category == 3) ? 3309 : 4627);
        };

        // SHA3 parameters
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::SHA3)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 1 || sec_category == 2 || sec_category == 4 || sec_category == 5, "SHA-3 only supports security categories 1, 2, 4, 5");

        public:
            static constexpr uint32_t capacity = sec_category == 1 ? 448 :
                                                 sec_category == 2 ? 512 :
                                                 sec_category == 4 ? 768 :
                                                                     1024;
            static constexpr uint32_t rate = 1600 - capacity;
            static constexpr uint32_t rate_8 = rate / 8;
            static constexpr uint32_t rate_32 = rate / 32;
            static constexpr uint32_t rate_64 = rate / 64;

            // 0000 0110 - For SHA3, input stream is suffixed with bits 01 followed by 10*1 padding
            static constexpr uint8_t pad = uint8_t(0x06);

            static constexpr bool has_digest_size = true;
            static constexpr size_t digest_size = capacity / (2 * 8) ; // capacity is double the digest
        };

        // SHAKE parameters
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::SHAKE)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 1 || sec_category == 2, "SHAKE only supports security categories 1, 2");

        public:
            static constexpr uint32_t capacity = 256 * sec_category;
            static constexpr uint32_t rate = 1600 - capacity;
            static constexpr uint32_t rate_8 = rate / 8;

            // 0001 1111 - For SHAKE, input stream is suffixed with bits 1111 followed by 10*1 padding
            static constexpr uint8_t pad = uint8_t(0x1F);

            static constexpr bool has_digest_size = false;
        };

        // SHA2
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::SHA2_32)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 1 || sec_category == 2, "SHA2_32 only supports security categories 1, 2");

        public:
            static constexpr uint32_t digest = sec_category == 1 ? 224 : 256;

            static constexpr bool has_digest_size = true;
            static constexpr size_t digest_size = digest / 8 ;
        };
        template<class... Operators>
        class pqc_full_description_helper<std::enable_if_t<has_algorithm<Operators...>(algorithm::SHA2_64)>, Operators...>
                                   : public pqc_description<Operators...> {

        private:
            static constexpr auto sec_category = pqc_description<Operators...>::this_pqc_security_category_v;
            static_assert(sec_category == 1 || sec_category == 2 || sec_category == 4 || sec_category == 5, "SHA2_64 only supports security categories 1, 2, 4, 5");

        public:
            static constexpr uint32_t digest = sec_category == 1 ? 224 : 
                                               sec_category == 2 ? 256 :
                                               sec_category == 4 ? 384 :
                                               512 ;

            static constexpr bool has_digest_size = true;
            static constexpr size_t digest_size = digest / 8 ;
        };
    } // namespace detail
} // namespace cupqc

#endif // CUPQC_DETAIL_PQC_DESCRIPTION_HPP

