#include "nr-transformer-interface.h"

#include <iostream>
#define UE_NUMS 2

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
    Ns3AiMsgInterfaceImpl<EnvStruct, ActStruct>* msgInterface =
        Ns3AiMsgInterface::Get()->GetInterface<EnvStruct, ActStruct>();

    assert(msgInterface->GetCpp2PyVector()->size() == UE_NUMS);
    
    msgInterface->CppSendBegin();
    for (int j = 0; j < UE_NUMS; ++j)
    {
        msgInterface->GetCpp2PyVector()->at(j).features =features[j];
    }
    msgInterface->CppSendEnd();
}
