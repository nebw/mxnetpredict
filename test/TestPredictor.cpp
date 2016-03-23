#include "MXNetPredictor.h"

#include <cmath>
#include <string>

#include <opencv2/core/core.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("MXNetPredictor", "[MXNetPredictor]") {
    // generate random input data
    const uint dim = 64;
    cv::Mat2f input(dim, dim);
    cv::RNG rng;
    rng.fill(input, cv::RNG::NORMAL, 0.5, 0.1);

    const std::string testdata_path(TESTDATA_PATH);
    const std::string a_symbol_path(testdata_path + "/filter-symbol.json");
    const std::string a_param_path(testdata_path + "/filter-0001.params");
    SECTION("setup") {
        GIVEN("a symbol path and a param path") {
            THEN("a MXNetPreditor can be constructed and safely destroyed") {
                REQUIRE_NOTHROW(mx::MXNetPredictor(a_symbol_path, a_param_path, dim, dim));
            }
        }
    }
    SECTION( "prediction" ) {
        GIVEN( "an image" ) {
            THEN("the predict function returns a valid float") {
                auto pred = mx::MXNetPredictor(a_symbol_path, a_param_path, dim, dim);
                REQUIRE_NOTHROW(pred.predict(input));
                const auto value = pred.predict(input);
                REQUIRE_FALSE(std::isnan(value));
            }
        }
    }
}
