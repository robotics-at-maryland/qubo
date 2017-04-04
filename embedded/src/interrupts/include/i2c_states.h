/* Ross Baehr
   R@M 2017
   ross.baehr@gmail.com
*/

//*****************************************************************************
// The states in the interrupt handler state machine.
//*****************************************************************************
#define STATE_IDLE              0
#define STATE_WRITE             1
#define STATE_WRITE_FINAL       2
#define STATE_WRITE_QUERY       3
#define STATE_WRITE_QUERY_FINAL 4
#define STATE_READ              5
#define STATE_READ_FINAL        6
