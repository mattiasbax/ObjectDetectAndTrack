#pragma once

#include <opencv2/core.hpp>

#include "CommonTypes.h"

class KalmanFilter {
   public:
    struct Parameters {
        Parameters(){};  // NOLINT(modernize-use-equals-default)
    };

    explicit KalmanFilter(Parameters&& parameters = Parameters()) : Param(std::move(parameters)) {}

   private:
    Parameters Param;
};