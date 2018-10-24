configuration BaseStationC {
  provides interface Intercept as RadioIntercept[am_id_t amid];
}
implementation {
  components MainC, BaseStationP, LedsC;
  components ActiveMessageC as Radio;

  RadioIntercept = BaseStationP.RadioIntercept;
  
  MainC.Boot <- BaseStationP;

  BaseStationP.RadioControl -> Radio;
 
  BaseStationP.RadioSend -> Radio;
  BaseStationP.RadioReceive -> Radio.Receive;
  BaseStationP.RadioSnoop -> Radio.Snoop;
  BaseStationP.RadioPacket -> Radio;
  BaseStationP.RadioAMPacket -> Radio;
  
  BaseStationP.Leds -> LedsC;
}
