#include "nr-transformer-interface.h"

using namespace ns3;

NrTransformerInterface::NrTransformerInterface()
{
    auto interface = Ns3AiMsgInterface::Get();
    interface->SetIsMemoryCreator(false);
    interface->SetUseVector(true);
    interface->SetHandleFinish(true);
}

NrTransformerInterface::~NrTransformerInterface()
{
}

TypeId
NrTransformerInterface::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::NrTransformerInterface").SetParent<Object>().SetGroupName("Ns3Ai").AddConstructor<NrTransformerInterface>();
    return tid;
}

void
NrTransformerInterface::SetFeatures(std::vector<FeaturesStruct>& features)
{
    Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>* msgInterface =
        Ns3AiMsgInterface::Get()->GetInterface<FeaturesStruct, ActStruct>();

    assert(msgInterface->GetCpp2PyVector()->size() == NUMS_UE);
    
    msgInterface->CppSendBegin();
    for (int j = 0; j < NUMS_UE; ++j)
    {
        msgInterface->GetCpp2PyVector()->at(j).qci =features[j].qci;
        msgInterface->GetCpp2PyVector()->at(j).priority = features[j].priority;
        msgInterface->GetCpp2PyVector()->at(j).holDelay = features[j].holDelay;
        msgInterface->GetCpp2PyVector()->at(j).delayBudget = features[j].delayBudget;
        msgInterface->GetCpp2PyVector()->at(j).avgThroughput = features[j].avgThroughput;
        msgInterface->GetCpp2PyVector()->at(j).potThroughput = features[j].potThroughput;
    }
    msgInterface->CppSendEnd();
}

std::vector<double>
NrTransformerInterface::GetWeight()
{
     Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>* msgInterface =
        Ns3AiMsgInterface::Get()->GetInterface<FeaturesStruct, ActStruct>();
    msgInterface->CppRecvBegin();
    std::vector<double> weights;
    for (int j = 0; j < NUMS_UE; ++j)
    {
        weights.push_back(msgInterface->GetPy2CppVector()->at(j).weight);
    }
    msgInterface->CppRecvEnd();
    return weights;
}
