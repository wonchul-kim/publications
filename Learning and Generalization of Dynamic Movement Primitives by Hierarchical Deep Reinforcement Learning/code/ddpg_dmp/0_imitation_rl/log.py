import logging
from time import sleep
import rospy

def main():
    LOG_PATH = '/home/icsl/catkin_ws/src/ddpg_ur3/scripts/Results/'
    
    for i in range(100):
        # logging.basicConfig(filename= LOG_PATH + 'myapp.log', level=logging.INFO)
        logging.basicConfig(level=logging.INFO)

        logging.info('Started')
        logging.info('Finished')
        string = str('sdflkj') + '\n' + str('sdlfkj')
        logging.info(string)
        a = 2
        logging.info('ddd' + str(a) + 'ssdf')
        
if __name__ == '__main__':
    # rospy.init_node('log')
    main()