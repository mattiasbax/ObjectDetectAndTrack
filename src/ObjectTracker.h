#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "CommonTypes.h"
#include "KalmanFilter.h"

class ObjectTracker {
   public:
    struct Parameters {
        Parameters(){};  // NOLINT(modernize-use-equals-default)
        unsigned int maxNumberOfTracks = 20U;
    };

    explicit ObjectTracker(Parameters&& parameters = Parameters()) : KalmanFilters(), Param(std::move(parameters)) {}

   private:
    Parameters Param;
    std::vector<KalmanFilter> KalmanFilters;
};