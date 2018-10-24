/*
 * Sphero.c
 *
 * Created: 2015-03-23 오후 1:33:31
 *  Author: chul
 */ 


#include <avr/io.h>
#include <avr/interrupt.h>

#define F_CPU 16000000UL

#define BAUD_DIV  (F_CPU/8/BAUD - 1)
#define BAUD_DIV_H BAUD_DIV >> 8
#define BAUD_DIV_L BAUD_DIV
#define BAUD 9600 // 115200 for JMOD-BT-1, 9600 for fb155bc, 38400 for HC-05, 9600 for HC-06

#include <util/delay.h>

#define TX_CH(ch, val) do { while(!(UCSR##ch##A & 0x20)); UDR##ch=val; } while(0)
#define RX_CH(ch, val) do { while(!(UCSR##ch##A & 0x80)); val = UDR##ch;  } while(0)
#define  AVAIL_RX(ch )  (UCSR##ch##A & 0x80)

unsigned int leftcnt=0;
unsigned int rightcnt=0;
unsigned int leftcurrentcnt=0;
unsigned int rightcurrentcnt=0;
unsigned int leftmode=0;
unsigned int oldleftmode=0;
unsigned int rightmode=0;
unsigned int oldrightmode=0;
unsigned int maxspeed = 200;
unsigned int anglecnt=0;
unsigned int angle =150;
	
ISR(TIMER0_OVF_vect)
{	
	leftcnt = (leftcnt < 400) ? ++leftcnt:0;
	rightcnt=leftcnt;	

	if(leftcnt==0)
	{
		

		switch(rightmode)
		{
			case 0:
				rightcurrentcnt=0;
			break;
			
			case 1:
			case 2:
				if(rightcurrentcnt< 400 -maxspeed) rightcurrentcnt++;
				else rightcurrentcnt--;
			break;
			
			case 3:
				if(rightcurrentcnt > 0) rightcurrentcnt--;
				else rightmode=2;
				break;

			case 4:
				if(rightcurrentcnt > 0) rightcurrentcnt--;
				else rightmode=1;
				break;

			case 5:
			case 6:
				if(rightcurrentcnt > 0) rightcurrentcnt--;
				else rightmode=0;
				break;
				
			case 7:
				break;
				
			case 8:				
				if(anglecnt<=angle){
					anglecnt++;
					if(rightcurrentcnt > 0) rightcurrentcnt--;
				}else rightmode=2;
				break;
			case 9:		
				if(anglecnt<=angle){
					anglecnt++;
					if(rightcurrentcnt > 0) rightcurrentcnt--;
				}else rightmode=1;
				break;
				
			
			default:
				rightcurrentcnt=0;
			break;
			
		}
		
		switch(leftmode)
		{
			case 0:
				leftcurrentcnt=0;
			break;
			
			case 1:
			case 2:
				if(leftcurrentcnt< 400 -maxspeed) leftcurrentcnt++;
				else leftcurrentcnt--;
			break;
			
			case 3:
				if(leftcurrentcnt > 0) leftcurrentcnt--;
				else leftmode=2;
				break;

			case 4:
				if(leftcurrentcnt > 0) leftcurrentcnt--;
				else leftmode=1;
				break;

			case 5:
			case 6:
				if(leftcurrentcnt > 0) leftcurrentcnt--;
				else leftmode=0;
				break;
				
			case 7:
				if(anglecnt<=angle){
					anglecnt++;
					if(leftcurrentcnt > 0) leftcurrentcnt--;
				}else leftmode=1;
				break;
			case 10:
				if(anglecnt<=angle){
					anglecnt++;
					if(leftcurrentcnt > 0) leftcurrentcnt--;
				}else leftmode=2;
				break;
			default:
				leftcurrentcnt=0;
			break;
			
		}
		
	}


}

int main(void)
{
    char ch;
	

    UCSR0A = 2; UCSR0B=0x18; UBRR0H=BAUD_DIV_H ; UBRR0L=BAUD_DIV_L;
    UCSR1A = 2; UCSR1B=0x18; UBRR1H=BAUD_DIV_H ; UBRR1L=BAUD_DIV_L;
	DDRE |= 0xFC;
	
	TIMSK |= _BV(TOV0);
	TCNT0=0x00;	
	TCCR0 |= _BV(CS00);
	sei();	
	
    while(1){
		
	    if(AVAIL_RX(1))
	    {
		    RX_CH(1, ch);
			
			switch(ch)
				{
					case 'q':	//정지
						switch(leftmode)
						{
							case 0:
							leftmode=0;	
							break;
								
							case 1:
							case 3:
							case 5:
							leftmode=5;
							break;
								
							case 2:
							case 4:
							case 6:
							leftmode=6;
							break;
							
							default:
							leftmode=0;
							break;								
						}
										
						switch(rightmode)
						{
							case 0:
							rightmode=0;	
							break;
								
							case 1:
							case 3:
							case 5:
							rightmode=5;
							break;
								
							case 2:
							case 4:
							case 6:
							rightmode=6;
							break;
							
							default:
							rightmode=0;
							break;								
						}

						break;
					case 'w':	//전진
						if(leftmode==0 || leftmode==1) leftmode=1;
						else leftmode=4;
						
						if(rightmode==0 || rightmode==1) rightmode=1;
						else rightmode=4;
						break;
						
					case 's':	//후진
						if(leftmode==0 || leftmode==2) leftmode=2;
						else leftmode=3;
						
						if(rightmode==0 || rightmode==2) rightmode=2;
						else rightmode=3;
						break;
						
					case 'a':	//좌회진
					
						anglecnt=0;
						if(rightmode==1 && leftmode==1) leftmode=7;
						if(rightmode==2 && leftmode==2) rightmode=8;
						break;
						
					case 'd':	//우회진

						anglecnt=0;
						if(rightmode==1  && leftmode==1) rightmode=9;
						if(rightmode==2  && leftmode==2) leftmode=10;
						break;
						
					case 'o':	//SPEED UP
						if(maxspeed>=50) maxspeed = maxspeed-10;
						else maxspeed=50;
						break;
						
					case 'l':	//SPEED DOWN
						if(maxspeed<=350) maxspeed = maxspeed+10;
						else maxspeed=350;						
						break;
					
					default:
						break;
				}
								
	    }

		switch(leftmode)
		{
			case 0:
				if(leftcurrentcnt==0) PORTE &= ~_BV(7);
;
			break;
			
			case 1:
			case 3:
			case 5:
			case 7:
			case 9:
				PORTE |=_BV(7);
				PORTE &= ~_BV(5);
				if(leftcnt<leftcurrentcnt) PORTE |=_BV(4);
				else PORTE &= ~_BV(4);
			break;
			
			case 2:
			case 4:
			case 6:
			case 8:
			case 10:
				PORTE |=_BV(7);
				PORTE &= ~_BV(4);
				if(leftcnt<leftcurrentcnt) PORTE |=_BV(5);
				else PORTE &= ~_BV(5);			
			break;
			
			default:
			break;
		}
		
		switch(rightmode)
		{
			case 0:
				if(rightcurrentcnt==0) PORTE &= ~_BV(6);
;
			break;
			
			case 1:
			case 3:
			case 5:
			case 7:
			case 9:
				PORTE |=_BV(6);
				PORTE &= ~_BV(3);
				if(rightcnt<rightcurrentcnt) PORTE |=_BV(2);
				else PORTE &= ~_BV(2);
			break;
			
			case 2:
			case 4:
			case 6:
			case 8:
			case 10:
				PORTE |=_BV(6);
				PORTE &= ~_BV(2);
				if(rightcnt<rightcurrentcnt) PORTE |=_BV(3);
				else PORTE &= ~_BV(3);			
			break;
			
			default:
			break;
		}

		
    }
}