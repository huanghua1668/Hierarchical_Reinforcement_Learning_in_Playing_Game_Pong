#######################################################################
# 06-27-18
# hhuang@math.fsu.edu
# object+rl for playing pong based on gym
# adam is not good for this one
# 08-20-18
# . predict the position of ball hit the agent paddle y location
# . stepStartMove with no extra 1 step version
# 09-28-18
# . fix the bug in getActionLastTwoSteps
# . in which position is wrongly shifted at stepToGo=2
# 10-02-18
# . do regression ahead 4 steps so as to try to reduct last step mse
# 10-15-18
# . add a_{t-1}
# . add a_{t-2}, a_{t-1}
# . add a_{t-3}, a_{t-2}, a_{t-1}
# 10-21-18
# . blow up after 400 epochs,= for action 0,0,2
#   half its learning rate
# 11-1-18
# . record the impinge effects
# 11-12-18
# . q(s,a) evaluation, s is v_y of ball, in [-6, 6]
# . a is the action bundles, such that impinge location - paddle
#   center equals [-4, 4]
# 11-15-18
# . test starting time effects
# . try to break if no score after 1000 frames


#######################################################################

from __future__ import print_function
import gym
from gym import wrappers, logger
import numpy as np
import csv

# all possible actions
ACTION_UP = 2
ACTION_DOWN = 3
NO_OP=0
# order is important
ACTIONS = [NO_OP, ACTION_UP, ACTION_DOWN]

# bound for position and velocity
# bounds work as a normalization of the frequencies
# by observing 1000 frames, ball x or y difference in contiguous
#     frames is within [-4, 4], paddle y difference is also within it. 
#0-1 for opponent, 0 for position difference in height
#                1 for position in height at frame I
#2-5 for ball, 2 for position difference in width
#              3 for position difference in height
#              4 for position in width
#              5 for position in height 
#6-7 for agent, 6 for position difference in height
#               7 for position in height at frame I
# PADDLE_MIN = -3.5
# PADDLE_MAX = 79+3.5
# PADDLE_DIFF_MIN=-8.
# PADDLE_DIFF_MAX=8.
# BALL_X_MAX = 79.
# BALL_X_MIN = 0.
# BALL_Y_MAX = 79.5
# BALL_Y_MIN = -0.5
# BALL_X_DIFF_MAX = 4.
# BALL_X_DIFF_MIN = -4.
# BALL_Y_DIFF_MAX = 4.
# BALL_Y_DIFF_MIN = -4.
#STATE_MIN=np.array([ -6., -6., 0.,  -0.5, -12, -3.5])
STATE_MIN=np.array([ -12, -12, -12, -1.5])
#STATE_MAX=np.array([ 6., 6., 79.,  79.5, 12, 82.5])
STATE_MAX=np.array([ 12, 12, 12, 81.5])

#domain constants
COLOR_OPPONENT=213
COLOR_AGENT=92
COLOR_BALL=236
HEIGHT_PADDLE=8
HEIGHT_BALL=2
WIDTH_BALL=1
# width of paddle is 2, and this is the x-position for the left point
X_POSITION_OPPONENT=8
X_POSITION_AGENT=70
DISCOUNT = 0.95

def getObjectPosition(I, height, color, xPosition=None):
    position=0.
    found=False
    if xPosition is not None:
        #for paddle
        # case: paddle might be not fully exposed
        if(I[0,xPosition]!=0):
            found=True
            for j in range(1,1+height):
                if(I[j,xPosition]!=color):
                    position=j-1-(height/2.0-0.5)
                    break
        elif(I[79,xPosition]!=0):
            found=True
            for j in range(78,78-height,-1):
                if(I[j,xPosition]!=color):
                    position=j+1+height/2.0-0.5
                    break
        else:
            for j in range(1,80):
                if(I[j,xPosition]==color):
                    found=True
                    position=j+height/2.0-0.5
                    break
        return found, position
    else:
        #for ball
        position=np.zeros(2, dtype=int)
        for i in range(80):
            if found:
                break
            else:
                for j in range(80):
                    if(I[i,j]==color):
                        found=True
                        position[0]=i
                        position[1]=j
                        break
        if not found:
            return found, np.zeros(2)
        if position[0]==0:
            if I[position[0]+1,position[1]]!=color:
                return found, np.array([position[1], position[0]-0.5])
            return found, np.array([ position[1], position[0]+0.5])
        elif position[0]==79:
            if I[position[0]-1,position[1]]!=color:
                return found, np.array([position[1], position[0]+0.5])
            return found, np.array([position[1], position[0]-0.5])
        return found, np.array([position[1], position[0]+0.5])

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #hhuang: 210 height, 160 width, 
    I = I[35:195] # crop
    #hhuang: I=I[35:195,:,:], and here 195 is not included
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    if 1==0:
        for i in range(80):
            for j in range(80):
                #if(I[i,j]!=0):
                #if(I[i,j]==213 or I[i,j]==92):
                if(I[i,j]==213):
                    print('I[',i,',',j,']=',I[i,j])
                    break
    position=np.zeros(3)
    #opponentFound, position[0]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_OPPONENT, X_POSITION_OPPONENT)
    ballFound, position[0:2]=getObjectPosition(I, HEIGHT_BALL, COLOR_BALL)
    agentFound, position[2]=getObjectPosition(I, HEIGHT_PADDLE, COLOR_AGENT, X_POSITION_AGENT)
    #found=ballFound and agentFound
    #print('position: ', position)
    #return found, position
    #return agentFound, (position[2]-STATE_MIN[0])/(STATE_MAX[0]-STATE_MIN[0])
    return ballFound, position[0:3]

def extract_state(newPosition, position, newFound=True, found=True):
    #position is the 6-d [dx_ball, dy_ball, x_ball, y_ball, dy_paddle, y_paddle]
    # newPosition is the 3d [x_ball, y_ball, y_paddle]
    #0-3 for ball, 0 for position difference in width
    #              1 for position difference in height
    #              2 for position in width
    #              3 for position in height 
    #4-5 for agent, 4 for position difference in height at frame I
    #               5 for position in height 

    velocityChanged=False
    if (newPosition[1]-position[3])!=position[1]: velocityChanged=True 
    position[1]=newPosition[1]-position[3]
    position[0]=newPosition[0]-position[2]
    position[4]=newPosition[2]-position[5]
    position[2:4]=newPosition[:2]
    position[5]=newPosition[2]
    return velocityChanged, position

class lms_position_prediction:
    def __init__(self, stepSize, dimension=4, order=20,
            beta2=0.999, dy=2):
        self.frame=1
        self.X=70.
        self.Y1=79.5
        self.Y0=-0.5
        self.dy=dy
        # resutls: state for v_y^ball, in [-6~6]
        #          outcomes for [-1, 0, 1], namely loss, score, bounce
        #          back
        self.results=np.zeros((13, 3))

        self.regression=regression(stepSize=stepSize, order=order)

    def output_results(self):
        self.results=self.results.astype(int)
        f=open('samples.dat', 'w')
        for i in range(13):
            out=(str(i-6)+' '+str(self.results[i,0])
                         +' '+str(self.results[i,1])
                         +' '+str(self.results[i,2])+'\n')
            f.write(out)
        f.close()

    def shift(self, position, newPosition):
        position[0]=position[1]
        position[1]=position[2]
        position[2]=newPosition-position[3]
        position[3]=newPosition
        return position

    def getActionLastSteps(self, stepToGo, position, y):
        #nextPos=np.zeros(3)
        state=(position-STATE_MIN)/(STATE_MAX-STATE_MIN)
        phi=self.regression.cal_features_fourier(state)
        #finalPos=np.zeros(np.power(3, stepToGo))
        action=np.random.choice(ACTIONS)
        predict=self.regression.value(phi, action)
        #print('action', index, 'predict', predict)
        return action, predict, phi

    def getAction(self, ballX, y, vFinal, position, paddleMoved=False):
        upSequence=np.array([81.5, 76.5, 66.5, 55.5, 44.5, 32.5, 
                             21.5,  9.5,  1.5, -0.5, -1.5 ])
        downSequence=np.array([-1.5,3.5, 13.5, 23.5, 35.5, 46.5,
                               58.5, 69.5, 78.5, 80.5, 81.5])
        delta=4.
        indexToArrive=0
        stepBack=False
        stepToGo=int((self.X-ballX)/2.+0.5)
        yAgent=position[3]
        dy=position[2]
        agentToMove=y-yAgent
        action=0
        lastStep=False
        #predict=None
        #phi=None
        #op=action
        if stepToGo<10:
            #if stepToGo<5:
            #print('frame: ', self.frame, ', ball x: ', ballX, ', stepToGo: ', 
            #        stepToGo, ', y: ',y, ', paddle to move ', agentToMove)
            #print('paddleMoved: ', paddleMoved)
            if paddleMoved==False:
                stepStartMove=int((abs(agentToMove)+4)/10.)+1
                # add 1 since the frequency of 1st dy be 0 is
                # otherwise too high 
                stepStartMove+=1
                if(yAgent==-1.5): stepStartMove+=1
                # test indicates if locates at -1.5, down will work as
                # [-1.5, 1.5, 10.5, 20.5, 32.5, 43.5, 55.5], 
                if stepToGo<=stepStartMove :
                    # in case updated y make the stepStartMove larger
                    # then stepToGo and not move yet
                    paddleMoved=True
                    if stepToGo<=4 and stepToGo>0:
                        action, predict, phi=self.getActionLastSteps(stepToGo, position, y)
                        if stepToGo==1:
                            lastStep=True
                        return action, predict, phi, paddleMoved, lastStep
                        #print('start move, stepToGo=2, op, dy, agentToMove: ',op, dy,
                        #    agentToMove)
                    # in case don't need to move, one move will be hazadous
                    else:
                        if agentToMove>0.:
                            action=ACTION_DOWN
                        else:
                            action=ACTION_UP
                        return action, paddleMoved
                    #print('move start from here, action, ball x: ', action, ballX)
            else:
                if agentToMove>0.:
                    action=ACTION_DOWN
                elif agentToMove<0.:
                    action=ACTION_UP
                else:
                    action=0
                
                if stepToGo<=4 and stepToGo>0:
                    # to adjust the case 2 more steps to go, and dy is large
                    # but agenttogo is too small, non op here
                    action, predict, phi=self.getActionLastSteps(stepToGo, position, y)
                    if stepToGo==1:
                        lastStep=True
                    return action, predict, phi, paddleMoved, lastStep
                    #print('stepToGo=2, op, dy, agentToMove: ',op, dy,
                    #        agentToMove)
                elif stepToGo==0:
                    if vFinal>=0.:
                        action=ACTION_UP
                        #action=ACTION_DOWN
                    else:
                        action=ACTION_DOWN
                        #action=ACTION_UP

                #print('moved, action: ', action)
        return action, paddleMoved

    #def predict(self, x0, y0, v, position):
    def predict(self, x0, y0, v):
        # y is the final position
        # X is the agent paddle location
        # Y1, Y0 is the lower / upper wall location
        # t for steps(might be non-integer) until arrive at X
        # y prediction for ball location when it arrive at X
        # vFinal ball dy when it arrive at X
        # dStep steps at which ball just hitted the wall and guarented to
        #    show up with the y-direction velocity change sign.
        X=self.X
        #if int(position[2])%2==1:
        if int(x0)%2==1:
            X+=1
        Y1=self.Y1
        Y0=self.Y0
        #x0=position[2]
        #y0=position[3]
        #v=position[1]
        #t=(X-position[2])/2.
        t=(X-x0)/2.
        #y=position[3]+v*t
        y=y0+v*t
        vFinal=np.copy(v)
        dStep=-1
        hitWall=False
        if(y<Y0 or y>Y1):
            dy=Y1-Y0
            if v>0.:
                remains=((X-x0)/2.*abs(v)-(Y1-y0))/dy
                order=int(remains)
                remains=(remains-int(remains))*dy
                if order % 2==0:
                    y=Y1-remains
                    vFinal=-vFinal
                else:
                    y=Y0+remains
            else:
                remains=((X-x0)/2.*abs(v)-(y0-Y0))/dy
                order=int(remains)
                remains=(remains-int(remains))*dy
                if order % 2==0:
                    y=Y0+remains
                    vFinal=-vFinal
                else:
                    y=Y1-remains
    
            done=False
            if v>0.:
                # add 3 just in case ball just experienced missing
                dStep=int((Y1-y0)/v)+3
            elif v<0.:
                dStep=int((y0-Y0)/abs(v))+3
            hitWall=True
            return hitWall, int(t+0.5), y, vFinal, dStep
        else:
            return hitWall, int(t+0.5), y, vFinal, dStep

class regression:
    def __init__(self, stepSize, dimension=4, order=20,
            beta2=0.999):
        self.pieces = np.power(order+1,dimension)
        self.order=order
        self.dimension=dimension
        self.c=np.zeros((self.pieces, self.dimension))
        self.cal_coeff_c()

        self.weights = np.zeros((3, 3, 3, 3, self.pieces))
        #self.stepSizes = np.ones(2*self.pieces)*stepSize
        self.stepSizes = np.ones(self.pieces)*stepSize
        self.cal_stepSize()
        self.frame=0
        self.loadW()

    def loadW(self, epo=1900):
        loaded=np.load('w_epo'+str(epo)+'_action0.npz')
        self.weights[0]=loaded['a']
        loaded=np.load('w_epo'+str(epo)+'_action2.npz')
        self.weights[1]=loaded['a']
        loaded=np.load('w_epo'+str(epo)+'_action3.npz')
        self.weights[2]=loaded['a']
        print('load weights at epo ', epo, 'successfully')

    def outputW(self, epo):
        np.savez_compressed('w_epo'+str(epo), a=self.weights)

    def cal_coeff_c(self):
        #c=np.zeros((pieces,dimension))
        coeff=[]
        for i in range(self.pieces):
            index=i
            c=np.zeros(self.dimension)
            for j in range(self.dimension):
                c[j]=index%(self.order+1)
                index //=(self.order+1)
            coeff.append(np.copy(c))
        piecesUpdated=len(coeff)
        self.c=np.vstack(coeff)
        self.pieces=piecesUpdated
            #print('c[',i,']=',c[i])
        #return c

    def cal_features_fourier(self, state):
        #phi=np.zeros(self.pieces)
        #for i in range(self.pieces):
        #    phi[i]=np.cos(np.pi*np.sum(state*self.c[i]))
        phi=np.cos(np.pi*np.sum(self.c*state, axis=1))
        return phi

    def cal_stepSize(self):
        for i in range(self.pieces):
            cNorm=np.sqrt(np.mean(self.c[i]*self.c[i]))
            if(cNorm!=0): self.stepSizes[i]/=cNorm

    # estimate the value of given state and action
    def value(self, phi, action3, action2, action1):
        index3=action3
        index2=action2
        index1=action1
        if action3>0: index3-=1
        if action2>0: index2-=1
        if action1>0: index1-=1
        index3=int(index3)
        index2=int(index2)
        index1=int(index1)
        values=np.zeros(3)
        values[0]=np.sum(self.weights[0, index3, index2, index1]*phi)
        values[1]=np.sum(self.weights[1, index3, index2, index1]*phi)
        values[2]=np.sum(self.weights[2, index3, index2, index1]*phi)
        return values

    # learn with given state, action and target
    def learn(self, action3, action2, action1, dw, epo):
        #for i in range(self.pieces):
            #print('phi[',i,']=',phi[i], 'newPhi[',i,']=',newPhi[i])
        #print('frequency', frequency)
        update=dw
        updateScale=np.linalg.norm(update)
        index3=action3
        index2=action2
        index1=action1
        if action3>0: index3-=1
        if action2>0: index2-=1
        if action1>0: index1-=1
        index3=int(index3)
        index2=int(index2)
        index1=int(index1)
        weightsScale=np.linalg.norm(self.weights[index3, index2, index1])
        if weightsScale==0.:
            print('epo: ', epo,  ' parame scale: ', weightsScale)
        else:
            print('epo: ', epo,  ' update scale/ parame scale: ',
                           updateScale/weightsScale)
        self.weights[index3, index2, index1] += update

def getAction(phi, actions, y, evaluator):
    ys=evaluator.regression.value(phi, actions[0], actions[1], actions[2])
    ys-=(y+evaluator.dy)
    index=np.argmin(np.absolute(ys))
    action=index
    if index>0:
        action+=1
    return action

def shift(actions, action):
    actions[0]=actions[1]
    actions[1]=actions[2]
    actions[2]=action

def play(evaluator, eps):
    env=gym.make('PongDeterministic-v0')
    observation=env.reset()
    #print(observation)
    done=False
    state=None
    rewardSum=0.
    found=False
    frame_init=evaluator.frame
    errors=0.
    counts=0
    records=[]
    vBall=0
    while not done:
        # game specific, neglect beginning and endding frames
        found=False
        position=None
        oldPosition=None
        vBall=None
        #positions=[dy_t-3, dy_t-2, dy_t-1, y_t]
        positions=np.zeros(4)
        actions=np.zeros(3)
        while not found:
            #print('not found')
            #action = np.random.choice(ACTIONS)
            action = NO_OP
            #env.render()
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            #print('frame', evaluator.frame)
            #evaluator.frame+=1
            newObservation, reward, done, info=env.step(action)
            found, oldPosition=prepro(newObservation)
            evaluator.shift(positions, oldPosition[2])
        newObservation, reward, done, info=env.step(NO_OP)
        found, position=prepro(newObservation)
        evaluator.shift(positions, position[2])
        if position[0]-oldPosition[0]<=0.:
            action = NO_OP
            while(position[0]-oldPosition[0]<=0.):
                oldPosition=np.copy(position)
                newObservation, reward, done, info=env.step(action)
                found, position=prepro(newObservation)
                evaluator.shift(positions, position[2])
            for i in range(4):
                oldPosition=np.copy(position)
                newObservation, reward, done, info=env.step(action)
                found, position=prepro(newObservation)
                evaluator.shift(positions, position[2])
        u=position[0]-oldPosition[0]
        v=position[1]-oldPosition[1]

        # now found, game begins
        hitWall, t, y, vFinal, dStep=evaluator.predict(position[0], position[1], v)
        state=(positions-STATE_MIN)/(STATE_MAX-STATE_MIN)
        # this phi is corresponding to state, no action 
        phi=evaluator.regression.cal_features_fourier(state)
        action=NO_OP
        #action = getAction(phi, actions, y, evaluator)

        oldV=v
        reward=0.
        frame0=evaluator.frame
        while reward==0.:
            oldPosition=np.copy(position)
            #env.env.ale.saveScreenPNG(b'pong_'+str(1000+evaluator.frame)+'.png')
            #print('frame', evaluator.frame)
            #env.render()
            newObservation, reward, done, info=env.step(action)
            evaluator.frame+=1
            #when reward=-1 or 1, ball will disappear
            found, position=prepro(newObservation)
            evaluator.shift(positions, position[2])
            shift(actions, action)
            #game specific, when ball hit the upper wall, it will
            #disappear sometimes and then re-appear again
            if not found:
                #keep using the old ball width position and motion info
                position[:2]=oldPosition[:2]
                if u>0.:
                    position[0]+=2.
                else:
                    position[0]-=2.
            
            if t==1:
                vBall=int(max(min(v, 6),-6))

            u=position[0]-oldPosition[0]
            v=position[1]-oldPosition[1]
            xBall=position[0]
            yBall=position[1]
            #print('ball postion, u, paddle:', position[:2], u,
            #        position[2] )
            #if (position[0]<=evaluator.X+1 and u>0.) or position[0]>=68:
            if (position[0]<=evaluator.X+1 and u>0.) or position[0]>=66:
                #print(evaluator.frame, action, position)
                if v==oldV and xBall<evaluator.X+1 and u>1.:
                    hitWall, t, y, vFinal, dStep=evaluator.predict(xBall, yBall, v)
                oldV=v
                if y is not None and t>-1:
                    state=(positions-STATE_MIN)/(STATE_MAX-STATE_MIN)
                    phi=evaluator.regression.cal_features_fourier(state)
                    action=NO_OP
                    action = getAction(phi, actions, y, evaluator)
                    if t==0:
                        temp=(y-position[2])/83.
                        loss=-temp*temp
                        errors-=loss
                        counts+=1

                    oldPosition=np.copy(position)
            else:
                positions=np.zeros_like(positions)
                actions=np.zeros_like(actions)
                action=NO_OP
                y=None
                oldV=None
            t-=1

            if ((position[0]==evaluator.X-5 or 
                 position[0]==evaluator.X-6) and u>0 and
                (vBall is not None)): 
                evaluator.results[vBall+6, 1]+=1

            #if evaluator.frame-frame0>600 and u<0.:
            if u<0.:
                if np.random.binomial(1, 0.05)==1:
                    action=np.random.choice(ACTIONS)

            if evaluator.frame-frame_init>13000: break


        if vBall is not None:
            if reward==-1:
                evaluator.results[vBall+6, 0]+=1
            elif reward==1:
                evaluator.results[vBall+6, 2]+=1
        #else:
        #    evaluator.results[vBall+6, 1]+=1

        rewardSum+=reward
        print('reward: ', reward, ' scores: ', rewardSum)
        #once non-zero score happened, the ball will disappear, no longer found
        if evaluator.frame-frame_init>13000:
            done=True
            print('total frames move right in 1 episode exceed 6500!')

    if counts!=0: 
        errors/=counts
        errors=np.sqrt(errors)*83.
    out=str(eps)+' '+str(int(rewardSum))+'\n'
    f=open('rewards.dat', 'a')
    f.write(out)
    f.close()
    print('episode: ', eps, ', rewards: ', rewardSum)

def figure_order_effect():
    runs = 1
    #alphas = np.arange(1, 2) / 20000.0
    #alphas = [0.003, 0.01]
    alphas = [0.01]
    #lams = [0.99, 0.98, 0.9, 0.8, 0.7]
    orders=[10]
    dimension=4
    #logger.set_level(logger.INFO)
    #video_path='/home/hh/Dropbox/rl_pong/object_rl/next_position_regression/play/'
    #env=gym.make('Pong-v0')
    
    f=open('rewards.dat', 'w')
    f.close()

    #myfile1=open('final_position_predicts.csv','w')
    #writer1=csv.writer(myfile1)
    #env=wrappers.Monitor(env, directory=video_path, force=True)
    for orderInd, order in enumerate(orders):
        for alphaInd, alpha in enumerate(alphas):
            for run in range(runs):
                evaluator =lms_position_prediction (alpha, dimension, order=order)
                for eps in range(1000):
                    mses=play(evaluator, eps)
                evaluator.output_results()

if __name__ == '__main__':
    figure_order_effect()




