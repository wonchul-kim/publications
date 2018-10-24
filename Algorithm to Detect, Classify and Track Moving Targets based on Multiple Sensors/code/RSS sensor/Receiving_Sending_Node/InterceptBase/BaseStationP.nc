#include "AM.h"
#include "Serial.h"

module BaseStationP @safe() {
  uses {
    interface Boot;
    interface SplitControl as RadioControl;
   
    interface AMSend as RadioSend[am_id_t id];
    interface Receive as RadioReceive[am_id_t id];
    interface Receive as RadioSnoop[am_id_t id];
    interface Packet as RadioPacket;
    interface AMPacket as RadioAMPacket;

    interface Leds;
  }
  
  provides interface Intercept as RadioIntercept[am_id_t amid];
}

implementation
{
  enum {
    RADIO_QUEUE_LEN = 12
  };

  message_t  radioQueueBufs[RADIO_QUEUE_LEN];
  message_t  *radioQueue[RADIO_QUEUE_LEN];
  uint8_t    radioIn, radioOut;
  bool       radioBusy, radioFull;

  task void radioSendTask();

  void dropBlink() {
    call Leds.led2Toggle();
  }

  void failBlink() {
    call Leds.led2Toggle();
  }

  event void Boot.booted() {
    uint8_t i;

    for (i = 0; i < RADIO_QUEUE_LEN; i++)
      radioQueue[i] = &radioQueueBufs[i];
      radioIn = radioOut = 0;
      radioBusy = FALSE;
      radioFull = TRUE;

      call RadioControl.start();
  }

  event void RadioControl.startDone(error_t error) {
    if (error == SUCCESS) {
      radioFull = FALSE;
    }
  }

  event void RadioControl.stopDone(error_t error) {}

  uint8_t count = 0;

  message_t* receive(message_t* msg, void* payload,
		     uint8_t len, am_id_t id);
  
  event message_t *RadioSnoop.receive[am_id_t id](message_t *msg,
						    void *payload,
						    uint8_t len) {
    return receive(msg, payload, len, id);
  }
  
  event message_t *RadioReceive.receive[am_id_t id](message_t *msg,
						    void *payload,
						    uint8_t len) {
    return receive(msg, payload, len, id);
  }


////////////////////////////////////////////////////////////////////////////////////////////////
  message_t* receive(message_t *msg, void *payload, uint8_t len, am_id_t id) {
    message_t *ret = msg;
    
    if (!signal RadioIntercept.forward[id](msg,payload,len))
      return ret;

    atomic {
      if (!radioFull)
	{
	  ret = radioQueue[radioIn];
	  radioQueue[radioIn] = msg;

	  radioIn = (radioIn + 1) % RADIO_QUEUE_LEN;
	
	  if (radioIn == radioOut)
	    radioFull = TRUE;

	  if (!radioBusy)
	    {
	      post radioSendTask();
	      radioBusy = TRUE;
	    }
	}
      else
	dropBlink();
    }
    
    return ret;
  }

  uint8_t tmpLen;
  
 task void radioSendTask() {
    uint8_t len;
    am_id_t id;
    am_addr_t addr;
    message_t* msg;
    
    atomic
      if (radioIn == radioOut && !radioFull)
	{
	  radioBusy = FALSE;
	  return;
	}

    msg = radioQueue[radioOut];
    tmpLen = len = call RadioPacket.payloadLength(msg);
//   addr = call RadioAMPacket.destination(msg);
    addr = 0x0000;
    id = call RadioAMPacket.type(msg);

    if (call RadioSend.send[id](addr, radioQueue[radioOut], len) == SUCCESS)
      call Leds.led0Toggle();
    else
      {
	failBlink();
	post radioSendTask();
      }
  }

  event void RadioSend.sendDone[am_id_t id](message_t* msg, error_t error) {
    if (error != SUCCESS)
      failBlink();
    else
      atomic
	if (msg == radioQueue[radioOut])
	  {
	    if (++radioOut >= RADIO_QUEUE_LEN)
	      radioOut = 0;
	    if (radioFull)
	      radioFull = FALSE;
	  }
    
    post radioSendTask();
  }

  default event bool
  RadioIntercept.forward[am_id_t amid](message_t* msg,
				       void* payload,
				       uint8_t len) {
    return TRUE;
  }

}  
