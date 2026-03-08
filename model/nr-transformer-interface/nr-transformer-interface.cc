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

    assert(msgInterface->GetCpp2PyVector()->size() == MAX_NUM_FLOWS);
    
    msgInterface->CppSendBegin();
    for (int j = 0; j < MAX_NUM_FLOWS; ++j)
    {
        msgInterface->GetCpp2PyVector()->at(j).rnti =features[j].rnti;
        msgInterface->GetCpp2PyVector()->at(j).priority = features[j].priority;
        msgInterface->GetCpp2PyVector()->at(j).dcGbrFlag = features[j].dcGbrFlag;
        msgInterface->GetCpp2PyVector()->at(j).pfMetric = features[j].pfMetric;
        msgInterface->GetCpp2PyVector()->at(j).delayFactor = features[j].delayFactor;
    }
    msgInterface->CppSendEnd();
}

std::vector<double>
NrTransformerInterface::GetWeight()
{
     Ns3AiMsgInterfaceImpl<FeaturesStruct, ActStruct>* msgInterface =
        Ns3AiMsgInterface::Get()->GetInterface<FeaturesStruct, ActStruct>();
    
    assert(msgInterface->GetPy2CppVector()->size() == 1);
    msgInterface->CppRecvBegin();
    std::vector<double> allocation;
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym1);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym2);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym3);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym4);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym5);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym6);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym7);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym8);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym9);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym10);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym11);
    allocation.push_back(msgInterface->GetPy2CppVector()->at(0).sym12);
    msgInterface->CppRecvEnd();
    return allocation;
}
