#pragma once
#include <ns3/ai-module.h>
#include "ns3/core-module.h"

using namespace ns3;

#define NUMS_UE 2

struct FeaturesStruct
{
    uint8_t qci; 
    uint8_t priority;
    uint16_t holDelay; 
    uint16_t delayBudget;
    uint16_t avgThroughput;
    uint16_t potThroughput;
};

struct EnvStruct
{
    FeaturesStruct features;
};

struct ActStruct
{
    double weight;
};

class NrTransformerInterface : public Object
{
    public:
    NrTransformerInterface();
    ~NrTransformerInterface() override;
    static TypeId GetTypeId();
    
    void SetFeatures(std::vector<FeaturesStruct>& features);
    std::vector<double> GetWeight();
};
