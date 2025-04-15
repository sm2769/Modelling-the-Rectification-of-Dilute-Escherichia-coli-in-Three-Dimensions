import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy import stats#
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams['figure.dpi'] = 300

#Constants T is tmperature, eta is viscosity, a is diameter of particle micrometre, kB boltzmann constant
T=294
eta=0.001
a=1
kB=1.386049e-23
 
#Diffusivity
D = ((kB*T)/(6*np.pi*eta*a))

#n = number of time steps, dimensions is number of dimensions, endtime is the end time 
n=10001
dimensions=3
endtime=1000
 
#Create set of times for use and find the individual time step dt
times = np.linspace(0., endtime, n)
dt = times[1] - times[0]

#Tumble angle
theta = np.pi/3

#Cell speed
speed=20

#box side length
ymin=-125
ymax=125
xmin=-500
xmax=500
zmin=-125
zmax=125

#Origin
#origin=np.array([250/2.0,0.0,0.0])

#trapping angle degrees
trapangle=60
freeangle=30

L=20


#Number of simulations
loops=1000


def x_rotation(vector,theta):
    #Rotates 3-D vector around x-axis
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    #Rotates 3-D vector around y-axis
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    #Rotates 3-D vector around z-axis
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def BrownianMotion():
     
    #Assigns a normally distributed number to an array of n size for given dimensions x,y,z
    #random.normal(loc=mean, scale=standard deviation, size= output)
    dB = np.random.normal(loc=0.0, scale=np.sqrt(2*D*dt), size=(n-1, dimensions))
    #Assigns starting positon of 0 toan array
    B0 = np.zeros(shape=(1, dimensions))
    #Adds arrays together and calculates the cumulative sum for each row
    global B
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)

    timeandcoord = np.column_stack((times, B))
    #Saves brownian motion array of coords to text file
    with open('BrownianMotion.txt', 'w') as f:
        for item in (timeandcoord):
            for i in item:
                f.write(str(i)+" ")
            f.write("\n")
    return B       
  
def MeanSquaredDsiplacement1D():    
    #Mean squared displacement array intialised with two columns all zeros of n/4 whole number length
    MSD = np.zeros(shape=((n//4)-1,2))
    #Initilising an array corresponding to 2*D*tau for comparison
    twoDtau = np.zeros(shape=((n//4)-1,2))
    B=BrownianMotion()
    
    #Calculating MSD for value of tau from 1 to n//4
    for tau in range(1,n//4):
        twoDtau[tau-1] =(tau, 2*D*tau*dt) 
        sumdiff=0
        
        #Calculation of MSD for given tau value
        for i in range(n-tau):
            sumdiff += ((B[i+tau,0])- (B[i,0]))**2
        MSD[tau-1] = (tau, (sumdiff/(n-tau)) )

    #temp = np.column_stack((MSD, twoDtau))
    #with open('MSD.txt', 'w') as filehandle:
        #json.dump(temp.tolist(), filehandle)    

    plt.plot(MSD[:,0],MSD[:,1], label ="MSD")
    plt.plot(twoDtau[:,0],twoDtau[:,1], label= "2Dtau")
    plt.title("Plot of Mean Squared Displacement for Tau Values")
    plt.xlabel(r"$\tau$")
    plt.ylabel("Mean Squared Displacement")
    plt.legend()
    plt.show()
    
def MeanSquaredDsiplacement3DBrownian():

    #Initialising mean and standard error on mean arrays
    mean = np.zeros(shape=((n//4)+1))
    sem = np.zeros(shape=((n//4)+1))
    #Variable for how many simlations of brownian motion to calcluate msd for
    nummsd=25
    #Mean squared displacement array intialised with nummsd columns all zeros of n/4 whole number length
    MSD = np.zeros(shape=((n//4)+1,nummsd))
    
    #Loop over for each simulation of brownian motion
    for counter in range(nummsd):
        
        #Initilising an array corresponding to 2*D*tau for comparison
        sixDtau = np.zeros(shape=((n//4)+1,2))
        #Calling the brownian motion function to fill an array with new values of brownian motion
        B = BrownianMotion()
        
        
        #Calculating MSD for value of tau from 1 to n//4
        for tau in range(1,(n//4)+1):
            #Filling 6*D*tau array with correct values for comparison
            sixDtau[tau] =(tau, 6*D*tau*dt) 
            
            #Resetting summ of the diffrences to 0
            sumdiff=0
            
            #Calculation of MSD for given tau value
            for i in range(n-tau):
                #Calculating running sum of the difference between two values
                sumdiff += ((B[i+tau,0])- (B[i,0]))**2+ ((B[i+tau,1])- (B[i,1]))**2+((B[i+tau,2])- (B[i,2]))**2
            
            #Filling first column of array with correct tau values
            MSD[tau,0] = (tau)
            #Filling column of array with MSD values
            MSD[tau,counter]= (sumdiff/(n-tau)) 
    
    #Calculating the mean for each coulumn
    for i in range(0,(n//4)+1):
        mean[i] = statistics.mean(MSD[i,1:])
    #Calculating the standard error on mean for each coulumn
    for i in range(0,(n//4)+1):
        sem[i] = stats.sem(MSD[i,1:]) 

    #temp = np.column_stack((MSD, twoDtau))
    #with open('MSD.txt', 'w') as filehandle:
        #json.dump(temp.tolist(), filehandle)    

    #Plotting
    #Plots Multiple Plots of MSDs Brownian Motion
    for k in range(1,nummsd):
        plt.plot(MSD[:,0]*dt,MSD[:,k])      
    plt.plot(sixDtau[:,0]*dt,sixDtau[:,1], label= "6Dtau")
    plt.title(f"Mean Squared Displacement Plot for Tau in 3D for {nummsd} Brownian Motion MSD plots")
    plt.xlabel(r"$\tau$ (s)", fontsize=15)
    plt.ylabel(r"MSD ($\mu$$m^2$)", fontsize=15)
    plt.savefig(f"MSDBrownianMultiplePlots time{endtime}s step{dt} numMSD{nummsd}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    #Plots Mean of Plots of MSDs Brownian Motion
    plt.errorbar(MSD[:,0]*dt,mean[:], yerr = sem, marker='o',linestyle='-',capsize=3, color="#648FFF",ecolor='#648FFF',mfc='#DC267F',label ="Mean of MSDs")
    plt.plot(sixDtau[:,0]*dt,sixDtau[:,1], color="#FE6100", label= "Analytical Solution")
    plt.title(f"Mean Squared Displacement Plot for Tau in 3D for {nummsd} Brownian Motion MSD plots")
    plt.xlabel(r"$\tau$ (s)", fontsize=15)
    plt.ylabel(r"MSD ($\mu$$m^2$)", fontsize=15)
    plt.legend()
    plt.savefig(f"MSDBrownianMean time{endtime}s step{dt} numMSD{nummsd}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
 
def MeanSquaredDsiplacement3DRunandTumble():

    #Initialising mean and standard error on mean arrays
    mean = np.zeros(shape=((n//4)+1))
    sem = np.zeros(shape=((n//4)+1))
    #Variable for how many simlations of brownian motion to calcluate msd for
    nummsd=25
    #Mean squared displacement array intialised with nummsd columns all zeros of n/4 whole number length
    MSD = np.zeros(shape=((n//4)+1,nummsd))
    print(dt)
    #Loop over for each simulation of rt
    for counter in range(nummsd):
        print("Status: ",counter+1," of ",nummsd)
        
        #Initilising an array corresponding to 2*D*tau for comparison
        #sixDtau = np.zeros(shape=((n//4)+1,2))
        #Calling the brownian motion function to fill an array with new values of brownian motion
        B = RunAndTumble()    
        
        #Calculating MSD for value of tau from 1 to n//4
        for tau in range(1,(n//4)+1):
            #Filling 6*D*tau array with correct values for comparison
            #sixDtau[tau] =(tau, 6*D*tau*dt) 
            
            #Resetting summ of the diffrences to 0
            sumdiff=0
            
            #Calculation of MSD for given tau value
            for i in range(n-tau):
                #Calculating running sum of the difference between two values
                sumdiff += ((B[i+tau,0])- (B[i,0]))**2+ ((B[i+tau,1])- (B[i,1]))**2+((B[i+tau,2])- (B[i,2]))**2
            
            #Filling first column of array with correct tau values
            MSD[tau,0] = (tau)
            #Filling column of array with MSD values
            MSD[tau,counter]= (sumdiff/(n-tau)) 
    
    #Calculating the mean for each coulumn
    for i in range(0,(n//4)+1):
        mean[i] = statistics.mean(MSD[i,1:])
    #Calculating the standard error on mean for each coulumn
    for i in range(0,(n//4)+1):
        sem[i] = stats.sem(MSD[i,1:])    

    

    plt.plot(MSD[:,0]*dt,mean[:], label ="Mean MSD",linewidth=2.5, color="#648FFF")
    D=((20**2)*1)/(3*0.5)
    x = np.linspace(0, 100000, 10000000)
    y = 6*D*x
    plt.plot(x,y, label =r"Dahlquist Result", linestyle='dashed', color="#DC267F")
    

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, 1E4); plt.ylim(1E2, 1E7)
    plt.title(f"Mean Squared Displacement for Tau in 3D for {nummsd} Run and Tumble MSD Plots", y=1.02)
    plt.xlabel(r"$\tau$ (s)", fontsize=15)
    plt.ylabel(r"MSD ($\mu$$m^2$)", fontsize=15)
    plt.legend()
    plt.savefig(f"MSDRunTumbleMean time{endtime}s step{dt} numMSD{nummsd} theta{theta}.pdf", format="pdf", bbox_inches="tight")
    plt.show()  
    
    
def UnitVector(previousunitvector, theta):
    Z = np.array([0,0,1])
    X = np.array([1,0,0])
    U = previousunitvector

    #Step (1) rotate the vector to lay on the z-axis
    #Find angle between xy projection and x-axis
    if U[0] == 0 and U[1] == 0:
        xangle=np.pi/2
    else:
        xangle = np.arccos(np.dot(np.array([U[0],U[1],0]) / np.linalg.norm(np.array([U[0],U[1],0])), X / np.linalg.norm(X)))
    #Rotate vector by above angle about z-axis so that it is on the x-axis
    Step1a = z_rotation(U, -xangle)
    #Calculate a new angle between vector and z-axis
    zangle = np.arccos(np.dot(Step1a / np.linalg.norm(Step1a), Z / np.linalg.norm(Z)))
    #Rotate vector by above angle about y-axis so that it lays on the z-axis
    Step1b = y_rotation(Step1a,-zangle)
    #Problem with arccos means that need to correct for negative angle
    if round(Step1b[2],10) !=1:
        xangle=-xangle
        Step1a = z_rotation(U, -xangle)
        Step1b = y_rotation(Step1a,-zangle)
    
    #Step (2)
    #Rotate vector about x-axis by 60 degrees
    Step2 = x_rotation(Step1b, theta)
    
    #Step (3)
    #Rotate vector about z-axis by random angle [0,360] degrees
    Step3 = z_rotation(Step2, np.random.uniform(0,2*np.pi))
    
    #Step (4)
    #Undo step 1
    Step4a = y_rotation(Step3, zangle)
    Step4b= z_rotation(Step4a, xangle)

    return (Step4b)

def RunAndTumble():
    #Initialising the array that will hold the values of postiion x,y,z
    RTcoordinates = np.zeros(shape=(n-1, dimensions))
    global initial
    initial = np.array([np.random.uniform(xmin+a,xmax-a),np.random.uniform(ymin+a,ymax-a),np.random.uniform(zmin+a,zmax-a)])
    RTcoordinates[0] = initial
    #print(RTcoordinates)
    #Defines a random vector pointing in any direction
    pointer = np.random.uniform(-1,1, size=3)
    unitvector = UnitVector(pointer/np.linalg.norm(pointer), theta)

    trapped=0

    #Starting a loop from 0 to the number of time steps decided at start of program
    for i in range(0,n-1):
        
        #Every second the particle will change in direction where i%(1/(endtime/(n-1))) is equal to zero every second
        if (i%(1/(endtime/(n-1))) == 0):
            previousunitvector=unitvector
            unitvector= UnitVector(previousunitvector, theta)
        
            temp=RTcoordinates[i-1]
            #if trapped true and angle of new vector with plane less than 30 then unitvecor = previous unit vector
            #x boundary
            if (trapped==1):
                #If the angle between new unit vector and boundary is more than free angle then free particle
                angle=abs(np.pi/2-np.arccos(unitvector[0]))
                if (angle>(freeangle*(np.pi/180))):
                    trapped=0
                #Otherwhise continue along surface with tumble
                else:
                    unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
            #y boundary
            elif (trapped==2): 
                #If the angle between new unit vector and boundary is more than free angle then free particle
                angle=abs(np.pi/2-np.arccos(unitvector[1]))
                if (angle>(freeangle*(np.pi/180))):
                    trapped=0
                #Otherwhise continue along surface with tumble                   
                else:
                    unitvector=np.array([unitvector[0],0,unitvector[2]])/ np.linalg.norm(np.array(([unitvector[0],0,unitvector[2]])))
            #z boundary
            elif (trapped==3):
                #If the angle between new unit vector and boundary is more than free angle then free particle
                angle=abs(np.pi/2-np.arccos(unitvector[2]))
                if (angle>(freeangle*(np.pi/180))):
                    #print(angle)
                    trapped=0
                #Otherwhise continue along surface with tumble 
                else:
                    unitvector=np.array([unitvector[0],unitvector[1],0])/ np.linalg.norm(np.array(([unitvector[0],unitvector[1],0])))
                    
            #Funnel boundary        
            #If x<l/2 then free
            elif (trapped==4):
                #Change
                angle=abs(np.pi/2-np.arccos(unitvector[0]))
                #Cahnge
                if (angle>(freeangle*(np.pi/180))) or temp[1]<L/2-a:
                    trapped=0
                else:
                #Change
                    unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
            
            elif (trapped==5):
                #Change
                angle=abs(np.pi/2-np.arccos(unitvector[0]))
                #Change
                if (angle>(freeangle*(np.pi/180))) or temp[1]>L/2-a:
                    trapped=0
                else:
                #Change
                    unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))

                                    
        #Calculate what the next position of the particle will be
        previouscoordinate=RTcoordinates[i-1]
        temp=RTcoordinates[i-1] + speed*dt*unitvector
        
        
        
        
        #If next position overlaps with the x boundary
        if (temp[0] - a <= xmin or temp[0] + a >= xmax) and ( trapped!=1):            
            #Calculate the angle between the yz plane and the particle direction
            #If less than trapping angle then move in parallel to yz plane and mark as trapped
            angle=abs(np.pi/2-np.arccos(unitvector[0]))            
            if (angle<(trapangle*(np.pi/180))):
                unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
                trapped=1
            #Otherwhise elastic collision with the x boundary
            else:
                unitvector[0]=unitvector[0]*-1
                
        #If next position overlaps with the y boundary
        if ((temp[1]- a) <= ymin or (temp[1] + a) >= ymax ) and ( trapped!=2):
            #Calculate the angle between the yz plane and the particle direction
            #If less than trapping angle then move in parallel to xz plane and mark as trapped
            angle=abs(np.pi/2-np.arccos(unitvector[1]))
            if trapped==4 or trapped==5:
                unitvector=np.array([-unitvector[0],-unitvector[1],unitvector[2]])
            elif (angle<(trapangle*(np.pi/180))) and (trapped!=4 and trapped!=5):
                unitvector=np.array([unitvector[0],0,unitvector[2]])/ np.linalg.norm(np.array(([unitvector[0],0,unitvector[2]])))
                trapped=2
            #Otherwhise elastic collision with the y boundary
            else:
                unitvector[1]=unitvector[1]*-1
        
        #If next position overlaps with the z boundary    
        if (temp[2] - a <= zmin or temp[2] + a >= zmax) and ( trapped!=3):
            #Calculate the angle between the yz plane and the particle direction
            #If less than trapping angle then move in parallel to xy plane and mark as trapped
            angle=abs(np.pi/2-np.arccos(unitvector[2])) 
            if (angle<(trapangle*(np.pi/180))) and (trapped!=4 and trapped!=5):
                unitvector=np.array([unitvector[0],unitvector[1],0])/np.linalg.norm(np.array([unitvector[0],unitvector[1],0]))
                trapped=3
            #Otherwhise elastic collision with the z boundary
            else:
                unitvector[2]=unitvector[2]*-1    
        
        #If the particle is within the region the funnel exists check for collision with the funnel
        #Change
        if (temp[1] >(L/2-a)) and (trapped!=4):
            #Check for collision with funnel equation
            #Change
            if (np.sign(temp[0]-a) != np.sign(previouscoordinate[0]-a) or np.sign(temp[0]+a) != np.sign(previouscoordinate[0]+a)):
                #Calculate the angle between the yz plane and the particle direction
                #If less than trapping angle then move in parallel to yz plane and mark as trapped
                #Change
                angle=abs(np.pi/2-np.arccos(unitvector[0])) 
                
                #left
                #Change
                if (temp[0])<0:                      
                    if (angle<(trapangle*(np.pi/180))):# and (trapped!=1 and trapped!=2 and trapped!=3):
                        #Change
                        unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
                        trapped=4
                    else:
                        #Elastic collision
                        #Cahnge
                        unitvector[0]=unitvector[0]*-1
                #right
                #Change
                elif (temp[0])>0:    
                    #Change
                    if (angle<(trapangle*(np.pi/180))):# and (trapped!=1 and trapped!=2 and trapped!=3):
                        unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
                        trapped=4
                    else:
                        #Elastic collision
                        #Change
                        unitvector[0]=unitvector[0]*-1
                    
        #Change   
        if (temp[1] <(-L/2+a)) and (trapped!=5):
            #Check for collision with funnel equation perp
            #Change
            if (np.sign(temp[0]-a) != np.sign(previouscoordinate[0]-a) or np.sign(temp[0]+a) != np.sign(previouscoordinate[0]+a)):
                #Calculate the angle between the yz plane and the particle direction
                #If less than trapping angle then move in parallel to yz plane and mark as trapped
                #Change
                angle=abs(np.pi/2-np.arccos(unitvector[0]))     
                
                #left
                #Change
                if (temp[0])<0:                
                    if (angle<(trapangle*(np.pi/180))):#and (trapped!=1 and trapped!=2 and trapped!=3):
                        #Change
                        unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
                        trapped=5                
                    else:
                        #Elastic collision
                        #Change
                        unitvector[0]=unitvector[0]*-1
                
                #right
                #Change
                if (temp[0])>0:                
                    if (angle<(trapangle*(np.pi/180))):#and (trapped!=1 and trapped!=2 and trapped!=3):
                        #Change
                        unitvector=np.array([0,unitvector[1],unitvector[2]])/ np.linalg.norm(np.array(([0,unitvector[1],unitvector[2]])))
                        trapped=5                
                    else:
                        #Elastic collision
                        #Change
                        unitvector[0]=unitvector[0]*-1

        #print(trapped)
        
        #Assigning a new position value using the last position plus the difference in distace in each coordinate based on speed and time step   
        RTcoordinates[i]= RTcoordinates[i-1] + speed*dt*unitvector
        
        if i==0:
            RTcoordinates[i]=initial+speed*dt*unitvector
            
    #Ensuring the initial position is at origin
    RTcoordinates = np.concatenate(([initial],RTcoordinates), axis=0)
        
    #print(RTcoordinates)                         
    return RTcoordinates

def MeanSquaredDisplacement():
    B = BrownianMotion()
    RTcoordinates = RunAndTumble()
    
    coordinates = B+RTcoordinates
    print(coordinates)

    #Plotting
    #Plots 3D
    ax = plt.axes(projection= '3d')
    #Adds centre point
    ax.scatter(0, 0, 0, c='g', marker='o')
    #Plots data
    ax.plot3D(coordinates[:,0], coordinates[:,1], coordinates[:,2], 'g', linewidth = '0.75')
    #Sets labels
    ax.set_title("Random Walk w 60 degree turns and Brownian Motion")#": n=",n," end time=",endtime," speed=",speed)
    ax.set_xlabel(r'X axis ($\mu$$m$)')#, fontsize=7.5)
    ax.set_ylabel(r'Y axis ($\mu$$m$)')#, fontsize=7.5)
    ax.set_zlabel(r'Z axis ($\mu$$m$)')#, fontsize=7.5)
    ax.set_box_aspect(None, zoom=0.8)
    ax.set_xlim(-500, 500); ax.set_ylim(-500, 500); ax.set_zlim(-500, 500);
    plt.show()

lastarray = np.zeros(shape=(loops,2))
counter=0
#Change
for i in range(loops):
    if i%10==0:
        print("Run: ",i)
    RTcoordinates=RunAndTumble()
    last=RTcoordinates[-1]
    lastarray[i,0]=last[0]
    lastarray[i,1]=last[1]
    if last[0]<0:
        counter+=1
print(counter/loops)

# #Plotting
# #Plots 3D
# # Create the figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# # Define the range of x and z values
# x = np.linspace(10, 125, 50)  # X range
# y = np.linspace(10, 125, 50)  # Y range
# z = np.linspace(-125, 125, 50)  # Z range

# # Create a meshgrid for x and z
# X, Z = np.meshgrid(x, z)
# Y, Z = np.meshgrid(y, z)



# # Plot the plane
# #Change
# ax.plot_surface(0, Y, Z, alpha=0.3, color='cyan')
# ax.plot_surface(0, -Y, Z, alpha=0.3, color='b')

# #Adds centre point
# ax.scatter(initial[0], initial[1], initial[2], c='g', marker='o')
# ax.scatter(last[0], last[1], last[2], c='r', marker='o')
# #Plots data
# ax.plot3D(RTcoordinates[:,0], RTcoordinates[:,1], RTcoordinates[:,2], c="g", linewidth = '0.75')
# #Sets labels
# #ax.set_title(r"Random Walk Trajectory of Particle Trapped in Box Sides Length 50$\mu$m")
# ax.set_xlabel(r'X axis ')#($\mu$$m$)')#, fontsize=7.5)
# ax.set_ylabel(r'Y axis ')#($\mu$$m$)')#, fontsize=7.5)
# ax.set_zlabel(r'Z axis ')#'($\mu$$m$)')#, fontsize=7.5)
# #ax.set_box_aspect(None, zoom=2)
# #ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax);
# ax.set_xlim(xmin, xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin, zmax);
# ax.set_box_aspect([abs(xmin)+abs(xmax), abs(ymin)+abs(ymax), abs(zmin)+abs(zmax)])
# ax.view_init(elev=90, azim=-90, roll=0)
# #ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
# plt.show()






bin_size = 10  # square bin of 10x10
x_bins = 1000 // bin_size
y_bins = 250 // bin_size
heatmap, xedges, yedges = np.histogram2d(lastarray[:,0], lastarray[:,1], bins=[x_bins, y_bins], range=[[xmin, xmax], [ymin, ymax]])

plt.figure(figsize=(10, 2.5))
plt.gca().set_aspect('equal')
#plt.figure()
#ax.invert_yaxis()
sns.heatmap(heatmap.T, cmap='hot', cbar=True, cbar_kws={"shrink": 0.5}, vmin=0, vmax=10, xticklabels=False,yticklabels=False)


#Funnel for 180
x_total = 500;y_total = 250;y1_start = 0;y1_end = 115;y2_start = 135;y2_end = 250;ymin1 = y1_start / y_total;ymax1 = y1_end / y_total;ymin2 = y2_start / y_total;ymax2 = y2_end / y_total
#Add two vertical lines at x=0 with a gap between
plt.axvline(x=50, ymin=ymin1, ymax=ymax1, linewidth=1, color="w")
plt.axvline(x=50, ymin=ymin2, ymax=ymax2, linewidth=1, color="w")

#plt.title(r"Particle Heat Map for Funnel Angle of 180$^\circ$", fontsize=15)
#ax.set_xlim(xmin, xmax); ax.set_ylim(ymin,ymax);
#ax.set_aspect(abs(ymin-ymax)/ abs(xmin-xmax))
plt.savefig("Heatmap 180.png", format="png", bbox_inches="tight")
plt.savefig("Heatmap 180.pdf", format="pdf", bbox_inches="tight")
plt.show()
