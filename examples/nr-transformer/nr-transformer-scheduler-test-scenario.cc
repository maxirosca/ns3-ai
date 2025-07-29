#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

// traffic model
// #include <ns3/abort.h>
// #include <ns3/config.h>
// #include <ns3/inet-socket-address.h>
// #include <ns3/internet-stack-helper.h>
// #include <ns3/ipv4-address-helper.h>
// #include <ns3/ipv4-global-routing-helper.h>
// #include <ns3/log.h>
// #include <ns3/packet-sink-helper.h>
// #include <ns3/packet-sink.h>
// #include <ns3/ping-helper.h>
// #include <ns3/simple-channel.h>
// #include <ns3/simple-net-device.h>
// #include <ns3/simulator.h>
// #include <ns3/traffic-generator-ftp-single.h>
// #include <ns3/traffic-generator-helper.h>
// #include <ns3/traffic-generator-ngmn-ftp-multi.h>
// #include <ns3/traffic-generator-ngmn-gaming.h>
// #include <ns3/traffic-generator-ngmn-video.h>
// #include <ns3/traffic-generator-ngmn-voip.h>
// #include <fstream>
// #include <ostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("NrTransformerSchedulerTestScenario");

Vector PlaceUeRandomly(double minDist, double maxDist, uint64_t& streamIndex)
{
    Ptr<UniformRandomVariable> angleRv = CreateObject<UniformRandomVariable>();
    Ptr<UniformRandomVariable> radiusRv = CreateObject<UniformRandomVariable>();

    angleRv->SetStream(streamIndex++);
    radiusRv->SetStream(streamIndex++);

    angleRv->SetAttribute("Min", DoubleValue(0.0));
    angleRv->SetAttribute("Max", DoubleValue(2 * M_PI));
    
    double theta = angleRv->GetValue();
    double r = std::sqrt(radiusRv->GetValue());
    double distance = minDist + r * (maxDist - minDist);

    double x = distance * std::cos(theta);
    double y = distance * std::sin(theta);

    NS_LOG_INFO("Placing UE at random position: (" << x << ", " << y << ")");
    return Vector(x, y, 1.5);
}
// enum TrafficType
// {
//     NGMN_FTP,
//     NGMN_VIDEO,
//     NGMN_GAMING,
//     NGMN_VOIP
// };

// TypeId
// GetTypeId(const TrafficType& item)
// {
//     switch (item)
//     {
//     case NGMN_FTP:
//         return TrafficGeneratorNgmnFtpMulti::GetTypeId();
//     case NGMN_VIDEO:
//         return TrafficGeneratorNgmnVideo::GetTypeId();
//     case NGMN_GAMING:
//         return TrafficGeneratorNgmnGaming::GetTypeId();
//     case NGMN_VOIP:
//         return TrafficGeneratorNgmnVoip::GetTypeId();
//     default:
//         NS_ABORT_MSG("Unknown traffic type");
//     };
// }

// std::string
// GetName(const TrafficType& item)
// {
//     switch (item)
//     {
//     case NGMN_FTP:
//         return "ftp";
//     case NGMN_VIDEO:
//         return "video";
//     case NGMN_GAMING:
//         return "gaming";
//     case NGMN_VOIP:
//         return "voip";
//     default:
//         NS_ABORT_MSG("Unknown traffic type");
//     };
// }

int
main(int argc, char* argv[])
{
    // Scenario parameters
    uint16_t gNbNum = 1;
    uint16_t ueNum = 2;
    bool logging = false;
   
    // Traffic parameters
    std::string scenario = "RMa"; // scenario
    uint32_t udpPacketSize = 3000;
    uint32_t lambda = 1000; // data rate = 24 Mbpbs
    // std::string ueMobilityModel = "ns3::ConstantPositionMobilityModel";
    // enum TrafficType trafficType = NGMN_FTP;
    
    // Simulation parameters
    Time simTime = MilliSeconds(1000);
    Time udpAppStartTime = MilliSeconds(400);
    uint64_t randomStream = 1;
    bool enableTransformer = true;

    // NR parameters
    uint16_t numerology = 0;
    double frequency = 4e9;
    double bandwidth = 20e6;
    double txPower = 43;
    std::string schedulerType = "Qos";
    uint8_t enableOfdma = 0; // 0: no OFDMA, 1: OFDMA
    uint8_t enableQoSLcScheduler = 0;
    uint16_t mcsTable = 2;

    // Where the output files are stored
    std::string simTag = "default";
    std::string outputDir = "./";
    
    
    
    CommandLine cmd(__FILE__);

    cmd.AddValue("gNbNum", "The number of gNbs in multiple-ue topology", gNbNum);
    cmd.AddValue("ueNum", "The number of in multiple-ue topology", ueNum);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.AddValue("packetSize",
                 "packet size in bytes",
                 udpPacketSize);
    cmd.AddValue("lambda",
                 "Number of UDP packets in one second",
                 lambda);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("numerology", "The numerology to be used", numerology);
    cmd.AddValue("centralFrequency", "The system frequency to be used", frequency);
    cmd.AddValue("bandwidth", "The system bandwidth to be used", bandwidth);
    cmd.AddValue("totalTxPower",
                 "Tx power to be used",
                 txPower);
    cmd.AddValue("simTag",
                 "tag to be appended to output filenames to distinguish simulation campaigns",
                 simTag);
    cmd.AddValue("outputDir", "directory where to store simulation results", outputDir);
    cmd.AddValue("ueLevelSchedulerType",
                 "Assign resources to an UE based on all its LCs. PF: Proportional Fair, "
                 "RR: Round-Robin, Qos: Quality of Service",
                 schedulerType);
    cmd.AddValue("enableOfdma",
                 "If set to true, it enables Ofdma scheduler. Default value is false (Tdma)",
                 enableOfdma);
    cmd.AddValue("scenario",
                 "The scenario for the simulation. Choose among 'RMa: Rural Macro', "
                 "'UMa: Urban Macro', 'UMi: Urban Micro'",
                 scenario);
    cmd.AddValue("enableLcLevelQos",
                 "If set to true, allocated bytes via UE-level scheduler are assigned to LCs based "
                 "on their QoS requirements. Default is Round-Robin (false)",
                 enableQoSLcScheduler);
    cmd.AddValue("randomStream",
                 "The random stream to be used for the simulation. Default is 1",
                 randomStream);
    cmd.AddValue("enableTransformer",
                 "If set to true, it enables the transformer for the scheduler. Default is false",
                 enableTransformer);
    // cmd.AddValue("ueMobilityModel",
    //              "Mobility model for the UEs",
    //              ueMobilityModel);

    cmd.Parse(argc, argv);

    /*
     * Check if the frequency is in the allowed range.
     * If you need to add other checks, here is the best position to put them.
     */
    NS_ABORT_IF(frequency < 0.5e9 && frequency > 100e9);

    /*
     * If the logging variable is set to true, enable the log of some components
     * through the code. The same effect can be obtained through the use
     * of the NS_LOG environment variable:
     *
     * export NS_LOG="UdpClient=level_info|prefix_time|prefix_func|prefix_node:UdpServer=..."
     *
     * Usually, the environment variable way is preferred, as it is more customizable,
     * and more expressive.
     */
    if (logging)
    {
         LogLevel logLevel1 =
            (LogLevel)(LOG_PREFIX_FUNC | LOG_PREFIX_TIME | LOG_PREFIX_NODE | LOG_LEVEL_INFO);
        LogComponentEnable("NrMacSchedulerNs3", logLevel1);
        LogComponentEnable("NrMacSchedulerTdma", logLevel1);
    }

    /*
     * In general, attributes for the NR module are typically configured in NrHelper.  However, some
     * attributes need to be configured globally through the Config::SetDefault() method. Below is
     * an example: if you want to make the RLC buffer very large, you can pass a very large integer
     * here.
     */
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));
    
    // create base stations and mobile terminals
    NodeContainer gnbNodes;
    NodeContainer ueNodes;
    gnbNodes.Create(gNbNum);
    ueNodes.Create(ueNum);
    
    ScenarioParameters scenarioParameters;
    if (scenario == "RMa")
    {
        scenarioParameters.m_bsHeight = 35;
        scenarioParameters.m_minBsUtDistance = 35;
        scenarioParameters.m_isd = 1732;
    }
    else if (scenario == "UMa")
    {
        scenarioParameters.m_bsHeight = 25;
        scenarioParameters.m_minBsUtDistance = 35;
        scenarioParameters.m_isd = 500;
    }
    else if (scenario == "UMi")
    {
        scenarioParameters.m_bsHeight = 10;
        scenarioParameters.m_minBsUtDistance = 10;
        scenarioParameters.m_isd = 200;
    }
    else
    {
        NS_ABORT_MSG("Unrecognized scenario: " << scenario);
    }

    // position the base stations
    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    gnbPositionAlloc->Add(Vector(0.0, 0.0, scenarioParameters.m_bsHeight)); 
    // for ( uint32_t j = 0; j < ueNum; j++)
    // {
    //     uint32_t a = 250;
    //     uint32_t b = 450;
    //     gnbPositionAlloc->Add(Vector(0.0 + j * a, 0.0 + j * b, hBS));
    // }
    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.SetPositionAllocator(gnbPositionAlloc);
    gnbMobility.Install(gnbNodes);

    Ptr<ListPositionAllocator> uePositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t j = 0; j < ueNum; j++)
    {
        Vector pos = PlaceUeRandomly(scenarioParameters.m_minBsUtDistance, scenarioParameters.m_isd, randomStream);
        uePositionAlloc->Add(pos);
    }
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.SetPositionAllocator(uePositionAlloc);
    ueMobility.Install(ueNodes);
    // // position the mobile terminals and enable the mobility
    // MobilityHelper uemobility;
    // if (ueMobilityModel == "ns3::RandomWalk2dMobilityModel")
    // {
    //     uemobility.SetMobilityModel(
    //         "ns3::RandomWalk2dMobilityModel",
    //         "Bounds", RectangleValue(Rectangle(0.0, 500.0, 0.0, 500.0)),  // Urban macro cell coverage
    //         "Speed", StringValue("ns3::UniformRandomVariable[Min=0.5|Max=5.0]"),  // UE mobility range
    //         "Direction", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=6.283184]"),
    //         "Mode", EnumValue(RandomWalk2dMobilityModel::MODE_TIME),
    //         "Time", TimeValue(Seconds(0.1))  // change direction every 3 seconds
    //     );
    // }
    // uemobility.SetMobilityModel(ueMobilityModel);
    // uemobility.Install(ueNodes);
    // for ( uint32_t j = 0; j < ueNum; j++)
    // {
    //     uint32_t a = 50;
    //     uint32_t b = 20;
    //     ueNodes.Get(j)->GetObject<MobilityModel>()->SetPosition(Vector(90 + j * a, 15 + j *b, hUT));
    // }
    
    /*
     * Create the scenario. In our examples, we heavily use helpers that setup
     * the gnbs and ue following a pre-defined pattern. Please have a look at the
     * GridScenarioHelper documentation to see how the nodes will be distributed.
     */
    // int64_t randomStream = 1;
    // GridScenarioHelper gridScenario;
    // gridScenario.SetRows(1);
    // gridScenario.SetColumns(gNbNum);
    // // All units below are in meters
    // gridScenario.SetHorizontalBsDistance(5.0);
    // gridScenario.SetVerticalBsDistance(5.0);
    // gridScenario.SetBsHeight(hBS);
    // gridScenario.SetUtHeight(hUT);
    // // must be set before BS number
    // gridScenario.SetSectorization(GridScenarioHelper::SINGLE);
    // gridScenario.SetBsNumber(gNbNum);
    // gridScenario.SetUtNumber(ueNum * gNbNum);
    // gridScenario.SetScenarioHeight(3); // Create a 3x3 scenario where the UE will
    // gridScenario.SetScenarioLength(3); // be distributed.
    // randomStream += gridScenario.AssignStreams(randomStream);
    // gridScenario.CreateScenario();

   /*
     * Create two different NodeContainer for the different traffic type.
     * In flow1Container we will put the UEs that will receive low-latency traffic,
     * while in flow2Container we will put the UEs that will receive the voice traffic.
     */
    NodeContainer flow1Container;
    NodeContainer flow2Container;
    
    for (uint32_t j = 0; j < ueNodes.GetN(); ++j)
    {
        Ptr<Node> ue = ueNodes.Get(j);
        j % 2 == 0 ? flow1Container.Add(ue) : flow2Container.Add(ue);
    }
    
    
    // NS_LOG_INFO("Creating " << gridScenario.GetUserTerminals().GetN() << " user terminals and "
    //                         << gridScenario.GetBaseStations().GetN() << " gNBs");

    /*
     * Setup the NR module
     * NR simulation:
     * - nrEpcHelper, which will setup the core network
     * - IdealBeamformingHelper, which takes care of the beamforming part
     * - NrHelper, which takes care of creating and connecting the various
     * part of the NR stack
     * - NrChannelHelper, which takes care of the spectrum channel
     */
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    // Put the pointers inside nrHelper
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);

    // Set the scheduler type
    std::stringstream scheduler;
    std::string subType;

    subType = !enableOfdma ? "Tdma" : "Ofdma";
    scheduler << "ns3::NrMacScheduler" << subType << schedulerType;
    std::cout << "Scheduler: " << scheduler.str() << std::endl;
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName(scheduler.str()));
    nrHelper->SetSchedulerAttribute("ActiveDlTransformer", BooleanValue(enableTransformer));
                                     

    // Set the scheduler type for the QoS LC scheduler if enabled
    if (enableQoSLcScheduler)
    {
        nrHelper->SetSchedulerAttribute("SchedLcAlgorithmType",
                                        TypeIdValue(NrMacSchedulerLcQos::GetTypeId()));
        std::cout << "QoS LC scheduler is enabled" << std::endl;
    }
     /*
     * Setup the configuration of the spectrum. One operation band is deployed
     * with 1 component carrier (CC), automatically generated by the ccBwpManager
     */
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1; 

    // Create channel API
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories(scenario, "Default", "ThreeGpp");
    auto bandMask = NrChannelHelper::INIT_PROPAGATION;

    // Set attributes for the channel
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));

    /*
     * The configured spectrum division for TDD is:
     *
     * |----Band1----|
     * |-----CC1-----|
     * |-----BWP1----|
     */

    // Create the configuration for the CcBwpHelper. SimpleOperationBandConf creates
    // a single BWP per CC
    CcBwpCreator::SimpleOperationBandConf bandConf(frequency,
                                                    bandwidth,
                                                    numCcPerBand);

    OperationBandInfo band= ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    // Assign the channel to the bands
    channelHelper->AssignChannelsToBands({band}, bandMask);
    allBwps = CcBwpCreator::GetAllBwps({band});
    
    Packet::EnableChecking();
    Packet::EnablePrinting();

    /*
     *  Attributes valid for all the nodes
     */
    // Beamforming method
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));

    // Core latency
    nrEpcHelper->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    // Antennas for all the UEs
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));

    // Antennas for all the gNbs
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<IsotropicAntennaModel>()));
    
    // Error Model: gNB and UE with same spectrum error model.
    std::string errorModel = "ns3::NrEesmIrT" + std::to_string(mcsTable);
    nrHelper->SetDlErrorModel(errorModel);
    nrHelper->SetUlErrorModel(errorModel);

    // Both DL and UL AMC will have the same model behind.
    nrHelper->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nrHelper->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));

    uint32_t bwpIdflow1 = 0;
    uint32_t bwpIdflow2 = 0;

    // gNb routing between Bearer and bandwidh part
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB",
                                                  UintegerValue(bwpIdflow1));
    nrHelper->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdflow2));

    // Ue routing between Bearer and bandwidth part
    nrHelper->SetUeBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(bwpIdflow1));
    nrHelper->SetUeBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(bwpIdflow2));

    /*
     * We have configured the attributes we needed. Now, install and get the pointers
     * to the NetDevices, which contains all the NR stack:
     */

    NetDeviceContainer gnbNetDev =
        nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer flow1NetDev = nrHelper->InstallUeDevice(flow1Container, allBwps);
    NetDeviceContainer flow2NetDev = nrHelper->InstallUeDevice(flow2Container, allBwps);
    
    NetDeviceContainer ueNetDevs(flow1NetDev);
    ueNetDevs.Add(flow2NetDev);

    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueNetDevs, randomStream);


    /*
     * Go node for node and change the attributes we have to setup
     * per-node.
     */

    // Get the first netdevice (gnbNetDev.Get (0)) and the first bandwidth part (0)
    // and set the attribute.
    nrHelper->GetGnbPhy(gnbNetDev.Get(0), 0)
        ->SetAttribute("Numerology", UintegerValue(numerology));
    nrHelper->GetGnbPhy(gnbNetDev.Get(0), 0)
        ->SetAttribute("TxPower", DoubleValue(txPower));
    

    
    // create the internet and install the IP stack on the UEs
    // get SGW/PGW and create a single RemoteHost
    Ptr<Node> pgw = nrEpcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // connect a remoteHost to pgw. Setup routing too
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.000)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);
    internet.Install(ueNodes);

    Ipv4InterfaceContainer ueflow1IpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(flow1NetDev));
    Ipv4InterfaceContainer ueflow2IpIface = nrEpcHelper->AssignUeIpv4Address(NetDeviceContainer(flow2NetDev));
    
    // attach UEs to the closest gNB
    nrHelper->AttachToClosestGnb(ueNetDevs, gnbNetDev);
    
    /*
     * Traffic part. Install two kind of traffic: low-latency and voice, each
     * identified by a particular source port.
     */
    uint16_t dlPortUeFlow1 = 1234;
    uint16_t dlPortUeFlow2 = 1235;

    ApplicationContainer serverApps;

    // The sink will always listen to the specified ports
    UdpServerHelper dlPacketSinkUeFlow1(dlPortUeFlow1);
    UdpServerHelper dlPacketSinkUeFlow2(dlPortUeFlow2);

    // The server, that is the application which is listening, is installed in the UE
    serverApps.Add(dlPacketSinkUeFlow1.Install(flow1Container));
    serverApps.Add(dlPacketSinkUeFlow2.Install(flow2Container));

    /*
     * Configure attributes for the different generators, using user-provided
     * parameters for generating a CBR traffic
     *
     * Flow1: Low-Latency configuration and object creation:
     */
    UdpClientHelper dlClientUeFlow1;
    dlClientUeFlow1.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientUeFlow1.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
    dlClientUeFlow1.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambda)));

    // The bearer that will carry low latency traffic
    NrEpsBearer flow1Bearer(NrEpsBearer::NGBR_LOW_LAT_EMBB);

    // The filter for the low-latency traffic
    Ptr<NrEpcTft> flow1Tft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter dlpfUeflow1;
    dlpfUeflow1.localPortStart = dlPortUeFlow1;
    dlpfUeflow1.localPortEnd = dlPortUeFlow1;
    flow1Tft->Add(dlpfUeflow1);

     /*
     * Flow2: Voice configuration and object creation:
     */
    UdpClientHelper dlClientUeFlow2;
    dlClientUeFlow2.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    dlClientUeFlow2.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
    dlClientUeFlow2.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambda)));

    // The bearer that will carry voice traffic
    NrEpsBearer flow2Bearer(NrEpsBearer::GBR_CONV_VOICE);

    // The filter for the voice traffic
    Ptr<NrEpcTft> flow2Tft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter dlpfUeflow2;
    dlpfUeflow2.localPortStart = dlPortUeFlow2;
    dlpfUeflow2.localPortEnd = dlPortUeFlow2;
    flow2Tft->Add(dlpfUeflow2);

    /*
     * Let's install the applications!
     */
    ApplicationContainer clientApps;
    
    for (uint32_t i = 0; i < flow1Container.GetN(); ++i)
    {
        Ptr<NetDevice> ueDevice = flow1NetDev.Get(i);
        Address ueAddress = ueflow1IpIface.GetAddress(i);

        // The client, who is transmitting, is installed in the remote host,
        // with destination address set to the address of the UE
        dlClientUeFlow1.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortUeFlow1)));
        // dlClient.SetAttribute("RemoteAddress", AddressValue(ueAddress));
        clientApps.Add(dlClientUeFlow1.Install(remoteHost));

        // Activate a dedicated bearer for the traffic type
        nrHelper->ActivateDedicatedEpsBearer(ueDevice, flow1Bearer, flow1Tft);
    }
    
    for (uint32_t i = 0; i < flow2Container.GetN(); ++i)
    {
        Ptr<NetDevice> ueDevice = flow2NetDev.Get(i);
        Address ueAddress = ueflow2IpIface.GetAddress(i);

        // The client, who is transmitting, is installed in the remote host,
        // with destination address set to the address of the UE
        dlClientUeFlow2.SetAttribute(
            "Remote",
            AddressValue(addressUtils::ConvertToSocketAddress(ueAddress, dlPortUeFlow2)));
        // dlClient.SetAttribute("RemoteAddress", AddressValue(ueAddress));
        clientApps.Add(dlClientUeFlow2.Install(remoteHost));

        // Activate a dedicated bearer for the traffic type
        nrHelper->ActivateDedicatedEpsBearer(ueDevice, flow2Bearer, flow2Tft);
    }

    // start UDP server and client apps
    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);

    // enable the traces provided by the nr module
    // nrHelper->EnableTraces();

    FlowMonitorHelper flowmonHelper;
    NodeContainer endpointNodes;
    endpointNodes.Add(remoteHost);
    endpointNodes.Add(ueNodes);

    Ptr<ns3::FlowMonitor> monitor = flowmonHelper.Install(endpointNodes);
    monitor->SetAttribute("DelayBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("JitterBinWidth", DoubleValue(0.001));
    monitor->SetAttribute("PacketSizeBinWidth", DoubleValue(20));

    Simulator::Stop(simTime);
    Simulator::Run();

    /*
     * To check what was installed in the memory, i.e., BWPs of gNB Device, and its configuration.
     * Example is: Node 1 -> Device 0 -> BandwidthPartMap -> {0,1} BWPs -> NrGnbPhy -> Numerology,
    GtkConfigStore config;
    config.ConfigureAttributes ();
    */

    // Print per-flow statistics
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());
    FlowMonitor::FlowStatsContainer stats = monitor->GetFlowStats();

    double averageFlowThroughput = 0.0;
    double averageFlowDelay = 0.0;

    std::ofstream outFile;
    std::string filename = outputDir + "/" + simTag;
    outFile.open(filename.c_str(), std::ofstream::out | std::ofstream::trunc);
    if (!outFile.is_open())
    {
        std::cerr << "Can't open file " << filename << std::endl;
        return 1;
    }

    outFile.setf(std::ios_base::fixed);

    double flowDuration = (simTime - udpAppStartTime).GetSeconds();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin();
         i != stats.end();
         ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        std::stringstream protoStream;
        protoStream << (uint16_t)t.protocol;
        if (t.protocol == 6)
        {
            protoStream.str("TCP");
        }
        if (t.protocol == 17)
        {
            protoStream.str("UDP");
        }
        outFile << "Flow " << i->first << " (" << t.sourceAddress << ":" << t.sourcePort << " -> "
                << t.destinationAddress << ":" << t.destinationPort << ") proto "
                << protoStream.str() << "\n";
        outFile << "  Tx Packets: " << i->second.txPackets << "\n";
        outFile << "  Tx Bytes:   " << i->second.txBytes << "\n";
        outFile << "  TxOffered:  " << i->second.txBytes * 8.0 / flowDuration / 1000.0 / 1000.0
                << " Mbps\n";
        outFile << "  Rx Bytes:   " << i->second.rxBytes << "\n";
        if (i->second.rxPackets > 0)
        {
            // Measure the duration of the flow from receiver's perspective
            averageFlowThroughput += i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000;
            averageFlowDelay += 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets;

            outFile << "  Throughput: " << i->second.rxBytes * 8.0 / flowDuration / 1000 / 1000
                    << " Mbps\n";
            outFile << "  Mean delay:  "
                    << 1000 * i->second.delaySum.GetSeconds() / i->second.rxPackets << " ms\n";
            // outFile << "  Mean upt:  " << i->second.uptSum / i->second.rxPackets / 1000/1000 << "
            // Mbps \n";
            outFile << "  Mean jitter:  "
                    << 1000 * i->second.jitterSum.GetSeconds() / i->second.rxPackets << " ms\n";
        }
        else
        {
            outFile << "  Throughput:  0 Mbps\n";
            outFile << "  Mean delay:  0 ms\n";
            outFile << "  Mean jitter: 0 ms\n";
        }
        outFile << "  Rx Packets: " << i->second.rxPackets << "\n";
    }

    double meanFlowThroughput = averageFlowThroughput / stats.size();
    double meanFlowDelay = averageFlowDelay / stats.size();

    outFile << "\n\n  Mean flow throughput: " << meanFlowThroughput << "\n";
    outFile << "  Mean flow delay: " << meanFlowDelay << "\n";

    outFile.close();

    std::ifstream f(filename.c_str());

    if (f.is_open())
    {
        std::cout << f.rdbuf();
    }

    Simulator::Destroy();

    if (argc == 0)
    {
        double toleranceMeanFlowThroughput = 0.0001 * 56.258560;
        double toleranceMeanFlowDelay = 0.0001 * 0.553292;

        if (meanFlowThroughput >= 56.258560 - toleranceMeanFlowThroughput &&
            meanFlowThroughput <= 56.258560 + toleranceMeanFlowThroughput &&
            meanFlowDelay >= 0.553292 - toleranceMeanFlowDelay &&
            meanFlowDelay <= 0.553292 + toleranceMeanFlowDelay)
        {
            return EXIT_SUCCESS;
        }
        else
        {
            return EXIT_FAILURE;
        }
    }
    else if (argc == 1 and ueNum == 9) // called from examples-to-run.py with these parameters
    {
        double toleranceMeanFlowThroughput = 0.0001 * 47.858536;
        double toleranceMeanFlowDelay = 0.0001 * 10.504189;

        if (meanFlowThroughput >= 47.858536 - toleranceMeanFlowThroughput &&
            meanFlowThroughput <= 47.858536 + toleranceMeanFlowThroughput &&
            meanFlowDelay >= 10.504189 - toleranceMeanFlowDelay &&
            meanFlowDelay <= 10.504189 + toleranceMeanFlowDelay)
        {
            return EXIT_SUCCESS;
        }
        else
        {
            return EXIT_FAILURE;
        }
    }
    else
    {
        return EXIT_SUCCESS; // we dont check other parameters configurations at the moment
    }
}
