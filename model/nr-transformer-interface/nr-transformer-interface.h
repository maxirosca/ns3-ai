#pragma once
#include <ns3/ai-module.h>
#include "ns3/core-module.h"

using namespace ns3;

#define MAX_NUM_FLOWS 5

struct FeaturesStruct
{
    uint8_t rnti; 
    uint8_t resource_type;
    uint16_t priority; 
    uint16_t packetDelayBudget;
    uint16_t queueSize;
    uint16_t availableSymbols;
    uint8_t mcs;
};

struct EnvStruct
{
    FeaturesStruct features;
};

struct ActStruct
{
    double sym1;
    double sym2;
    double sym3;
    double sym4;
    double sym5;
    double sym6;
    double sym7;
    double sym8;
    double sym9;
    double sym10;
    double sym11;
    double sym12;
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
