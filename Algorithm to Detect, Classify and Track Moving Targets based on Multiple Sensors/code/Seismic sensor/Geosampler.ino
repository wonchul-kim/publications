/****************************************************************************
 * Record geophone data and output the data (from 0 through 4095) to the serial port.
 ***************************************************************************/

/* Serial speed for the report generation.  It should be fast enough to
   allow several values to be passed per second.  A speed of 38,400 baud
   should suffice for worst case reports of about 2,600 bytes per second. */
#define SERIAL_SPEED    9600

#define GEODATA_PIN          0

#define REPORT_BLINK_ENABLED   1
#define REPORT_BLINK_LED_PIN  13

#define SAMPLE_RATE   512

#define LED_PIN             13

/* Create a double buffer for geodata samples. */
#define NUMBER_OF_GEODATA_SAMPLES 256
short geodata_samples[ NUMBER_OF_GEODATA_SAMPLES * 2 ];
short *geodata_samples_real;
/* Indexes used by the interrupt service routine. */
int  isr_current_geodata_index;
/* Semaphor indicating that a frame of geophone samples is ready. */
bool geodata_buffer_full;
/* Flag that indicates that a report with amplitude information was
   created.  It is used by the report LED blinking. */
bool report_was_created;

float sensorValue;

/**
 * Setup the timer interrupt and prepare the geodata sample buffers for
 * periodic sampling.  Timer1 is used to generate interrupts at a rate of
 * 512 Hz.
 */
void start_sampling( )
{
  isr_current_geodata_index = 0;
  geodata_buffer_full       = false;

  // Set timer1 interrupt to sample at 512 Hz. */
  const unsigned short prescaling     = 1;
  const unsigned short match_register = F_CPU / ( prescaling * SAMPLE_RATE ) - 1;
  cli( );
  TCCR1B = ( TCCR1B & ~_BV(WGM13) ) | _BV(WGM12);
  TCCR1A = TCCR1A & ~( _BV(WGM11) | _BV(WGM10) );
  TCCR1B = ( TCCR1B & ~( _BV(CS12) | _BV(CS11) ) ) | _BV(CS10);
  OCR1A = match_register;
  TIMSK1 |= _BV(OCIE1A);
  sei( );
}

/**
 * Interrupt service routine for Arduino Mega devices which invokes the
 * generic interrupt service routine.
 */
ISR(TIMER1_COMPA_vect)
{
  sampling_interrupt( );
}


/*
 * Interrupt service routine for sampling the geodata.  The geodata analog
 * pin is sampled at each invokation of the ISR.  If the buffer is full, a
 * pointer is passed to the main program and a semaphor is raised to indicate
 * that a new frame of samples is available, and future samples are written
 * to the other buffer.
 *
 * While not a sampling task, we take advantage of the timer interrupt to
 * blink the report LED if enabled.
 */
void sampling_interrupt( )
{
  /* Read a sample and store it in the geodata buffer. */
  const int adc_resolution = 1024;

  short geodata_sample = analogRead( GEODATA_PIN ) - ( adc_resolution >> 1 );
  /* Scale the sample. */
  const int scale = 8192 / adc_resolution;
  geodata_sample = (short)( (double)geodata_sample * scale );
  geodata_samples[ isr_current_geodata_index++ ] = geodata_sample;

  /* Raise a semaphor if the buffer is full and tell which buffer
     is active. */
  if( isr_current_geodata_index == NUMBER_OF_GEODATA_SAMPLES )
  {
    geodata_samples_real     = &geodata_samples[ 0 ];
    geodata_buffer_full      = true;
  }
  else if( isr_current_geodata_index == NUMBER_OF_GEODATA_SAMPLES * 2 )
  {
    geodata_samples_real      = &geodata_samples[ NUMBER_OF_GEODATA_SAMPLES ];
    isr_current_geodata_index = 0;
    geodata_buffer_full       = true;
  }

  /* In the same interrupt routine, handle report LED blinking. */
  report_blink( REPORT_BLINK_ENABLED );
}



/**
 * Blink the report LED if it has been enabled.
 *
 * @param enabled @a true if report blinking has been enabled.
 */
void report_blink( bool enabled )
{
  static unsigned long timestamp;
  static bool          led_on = false;

  if( enabled == true )
  {
    /* Turn on the LED and start a timer if a report was created. */
    if( report_was_created == true )
    {
      report_was_created = false;
      timestamp = millis( ) + 50;
      digitalWrite( REPORT_BLINK_LED_PIN, HIGH );
      led_on = true;
    }
    /* Turn off the LED once the timer expires. */
    if( led_on == true )
    {
      if( millis( ) > timestamp )
      {
        digitalWrite( REPORT_BLINK_LED_PIN, LOW );
        led_on = false;
      }
    }
  }
}


/**
 * Send the samples in the most recent buffer over the serial port.
 *
 * @param [in] freq_real Array of samples.
 * @param [in] length Number of samples.
 */
void report( const short *samples, int length )
{
  /* Send all the samples in the buffer to the serial port. */
  for( int index = 0; index < length; index++ )
  {
    //Serial.print( " " );
    sensorValue = samples[index];
    Serial.println( sensorValue );
    //delay(100); // 1ms
  }
  /* Indicate to the report LED blinking that the report was submitted. */
  report_was_created = true;
}



/**
 * Initialize the serial port, setup the sampling, and turn off the on-board
 * LED.
 */
void setup()
{
  /* Initialize the serial port with the desired speed. */
  Serial.begin( SERIAL_SPEED );

  /* Setup the geophone data sampling buffers and sampling interrupt. */
  start_sampling( );

  /* Turn off the on-board LED. */
  pinMode( LED_PIN, OUTPUT );
  digitalWrite( LED_PIN, LOW );

  /* Configure the report LED if enabled. */
  report_was_created = false;
  if( REPORT_BLINK_ENABLED )
  {
    pinMode( REPORT_BLINK_LED_PIN, OUTPUT );
    digitalWrite( REPORT_BLINK_LED_PIN, LOW );
  }
}



/**
 * Main program loop which reports the samples every time the sample buffer
 * has been filled.
 */
void loop()
{
  /* Analyze the geophone data once it's available. */
  if( geodata_buffer_full == true )
  {
    geodata_buffer_full = false;

    /* Transmit the samples over the serial port. */
    report( geodata_samples, NUMBER_OF_GEODATA_SAMPLES );
  }
}

