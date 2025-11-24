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
    uint8_t sym1;
    uint8_t sym2;
    uint8_t sym3;
    uint8_t sym4;
    uint8_t sym5;
    uint8_t sym6;
    uint8_t sym7;
    uint8_t sym8;
    uint8_t sym9;
    uint8_t sym10;
    uint8_t sym11;
    uint8_t sym12;
};

class NrTransformerInterface : public Object
{
    public:
    NrTransformerInterface();
    ~NrTransformerInterface() override;
    static TypeId GetTypeId();
    
    void SetFeatures(std::vector<FeaturesStruct>& features);
    std::vector<uint8_t> GetWeight();
};
