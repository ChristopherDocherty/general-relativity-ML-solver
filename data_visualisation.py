import matplotlib.pyplot as plt


def save_loss_plot(filename, history):
    plt.yscale("log")
    plt.title("Full Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'])
    plt.savefig(filename)

def save_PDE_loss_plot(filename, history):

    plt.clf()
    plt.title("PDE Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(history.history['PDE_loss'])
    plt.savefig(filename)

def save_BC_loss_plot(filename, history):

    plt.clf()
    plt.title("BC Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(history.history['BC_loss'])
    plt.savefig(filename)


def save_fixed_point_plots(filename, history):


    fixed_point_results = {key: value for key, value in history.history.items() if key not in { "loss", "PDE_loss", "BC_loss"} }

    num_fp = len(fixed_point_results)

    
    plt.figure(1, (6.5 * num_fp,6))
    plt.clf()


    for i, (radius, fp_values) in enumerate(fixed_point_results.items()):

        plt.subplot(1,num_fp + 1, i+1)
        plt.title(f"r = {radius}")
        plt.xlabel("Samples (x1000)")
        plt.ylabel(r"$g_rr$")
        plt.plot(fp_values)
    
    plt.savefig(filename)

def save_losses_plot(filename, history):


#    plt.rcParams["figure.figsize"] = (20,10)
    plt.figure(1, (20,6))
    plt.clf()

    plt.subplot(1,3,1)
    plt.yscale("log")
    plt.title("Full Loss")
    plt.xlabel("Samples (x1000)")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'])


    plt.subplot(1,3,2)
    plt.title("PDE Loss")
    plt.xlabel("Samples (x1000)")
    plt.ylabel("Loss")
    plt.plot(history.history['PDE_loss'])


    plt.subplot(1,3,3)
    plt.title("BC Loss")
    plt.xlabel("Samples (x1000)")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(history.history['BC_loss'])

    plt.savefig(filename)


def save_grr_plot(filename, coords, minkowski, true_metric): 

    r = coords[:,1]

#    plt.figure(1, (10,10))
#    plt.clf()
#    plt.subplot(4,4,1)
#    plt.title(r'$g_{tt}$', fontsize=16)
#    plt.plot(r, minkowski[:,0,0], 'b', r, true_metric[:,0], 'r')
#
#    plt.subplot(4,4,2)
#    plt.title(r'$g_{tr}$', fontsize=16)
#    plt.plot(r, minkowski[:,0,1], 'b')
#
#    plt.subplot(4,4,3)
#    plt.title(r'$g_{t\theta}$', fontsize=16)
#    plt.plot(r, minkowski[:,0,2], 'b')
#
#    plt.subplot(4,4,4)
#    plt.title(r'$g_{t\phi}$', fontsize=16)
#    plt.plot(r, minkowski[:,0,3], 'b')
#
#
#    plt.subplot(4,4,5)
#    plt.title(r'$g_{rt}$', fontsize=16)
#    plt.plot(r, minkowski[:,1,0], 'b')
#
#    plt.subplot(4,4,6)
#    plt.title(r'$g_{rr}$', fontsize=16)
#    plt.plot(r, minkowski[:,1,1], 'b', r, true_metric[:,1], 'r')
#
#    plt.subplot(4,4,7)
#    plt.title(r'$g_{r\theta}$', fontsize=16)
#    plt.plot(r, minkowski[:,1,2], 'b') 
#
#    plt.subplot(4,4,8)
#    plt.title(r'$g_{r\phi}$', fontsize=16)
#    plt.plot(r, minkowski[:,1,3], 'b') 
#
#
#    plt.subplot(4,4,9)
#    plt.title(r'$g_{\theta t}$', fontsize=16)
#    plt.plot(r, minkowski[:,2,0], 'b') 
#
#    plt.subplot(4,4,10)
#    plt.title(r'$g_{\theta r}$', fontsize=16)
#    plt.plot(r, minkowski[:,2,1], 'b') 
#
#    plt.subplot(4,4,11)
#    plt.title(r'$g_{\theta \theta}$', fontsize=16)
#    plt.plot(r, minkowski[:,2,2], 'b', r, true_metric[:,2], 'r') 
#
#    plt.subplot(4,4,12)
#    plt.title(r'$g_{\theta\phi}$', fontsize=16)
#    plt.plot(r, minkowski[:,2,3], 'b') 
#
#
#    plt.subplot(4,4,13)
#    plt.title(r'$g_{\phi t}$', fontsize=16)
#    plt.plot(r, minkowski[:,3,0], 'b') 
#
#    plt.subplot(4,4,14)
#    plt.title(r'$g_{\phi r}$', fontsize=16)
#    plt.plot(r, minkowski[:,3,1], 'b') 
#
#    plt.subplot(4,4,15)
#    plt.title(r'$g_{\phi \theta}$', fontsize=16)
#    plt.plot(r, minkowski[:,3,2], 'b') 
#
#    plt.subplot(4,4,16)
#    plt.title(r'$g_{\phi \phi}$', fontsize=16)
#    plt.plot(r, minkowski[:,3,3], 'b', r , true_metric[:,3], 'r') 

#    plt.savefig(filename)


    plt.figure(1, (10,10))
    plt.clf()
    
    plt.subplot(1,3,1)
    plt.title(r'$g_{rr} predicted$')
    plt.xlabel(r'$r$')
    plt.plot(r, minkowski[:,1,1], 'b')


    plt.subplot(1,3,2)
    plt.title(r'$g_{rr} actual$')
    plt.xlabel(r'$r$')
    plt.plot(r, true_metric[:,1], 'r')


    plt.subplot(1,3,3)
    plt.title(r'$g_{rr} both$')
    plt.xlabel(r'$r$')
    plt.plot(r, minkowski[:,1,1], 'b', r, true_metric[:,1], 'r')
    
    plt.savefig(filename)


def save_results_plot(filename, coords, results, true_metric):
    
    r = coords[:,1]


    print(results)


    plt.rcParams["figure.figsize"] = (20,20)
    plt.figure(1)
    plt.clf()
    plt.subplot(4,4,1)
    plt.title(r'$g_{tt}$', fontsize=16)
    plt.plot(r, results[:,0], 'b', r, true_metric[:,0], 'r')

    plt.subplot(4,4,2)
    plt.title(r'$g_{tr}$', fontsize=16)
    plt.plot(r, results[:,1], 'b')

    plt.subplot(4,4,3)
    plt.title(r'$g_{t\theta}$', fontsize=16)
    plt.plot(r, results[:,2], 'b')

    plt.subplot(4,4,4)
    plt.title(r'$g_{t\phi}$', fontsize=16)
    plt.plot(r, results[:,3], 'b')


    plt.subplot(4,4,5)
    plt.title(r'$g_{rt}$', fontsize=16)
    plt.plot(r, results[:,1], 'b')

    plt.subplot(4,4,6)
    plt.title(r'$g_{rr}$', fontsize=16)
    plt.plot(r, results[:,4], 'b', r, true_metric[:,1], 'r')

    plt.subplot(4,4,7)
    plt.title(r'$g_{r\theta}$', fontsize=16)
    plt.plot(r, results[:,5], 'b') 

    plt.subplot(4,4,8)
    plt.title(r'$g_{r\phi}$', fontsize=16)
    plt.plot(r, results[:,6], 'b') 


    plt.subplot(4,4,9)
    plt.title(r'$g_{\theta t}$', fontsize=16)
    plt.plot(r, results[:,2], 'b') 

    plt.subplot(4,4,10)
    plt.title(r'$g_{\theta r}$', fontsize=16)
    plt.plot(r, results[:,5], 'b') 

    plt.subplot(4,4,11)
    plt.title(r'$g_{\theta \theta}$', fontsize=16)
    plt.plot(r, results[:,7], 'b', r, true_metric[:,2], 'r') 

    plt.subplot(4,4,12)
    plt.title(r'$g_{\theta\phi}$', fontsize=16)
    plt.plot(r, results[:,8], 'b') 


    plt.subplot(4,4,13)
    plt.title(r'$g_{\phi t}$', fontsize=16)
    plt.plot(r, results[:,3], 'b') 

    plt.subplot(4,4,14)
    plt.title(r'$g_{\phi r}$', fontsize=16)
    plt.plot(r, results[:,6], 'b') 

    plt.subplot(4,4,15)
    plt.title(r'$g_{\phi \theta}$', fontsize=16)
    plt.plot(r, results[:,8], 'b') 

    plt.subplot(4,4,16)
    plt.title(r'$g_{\phi \phi}$', fontsize=16)
    plt.plot(r, results[:,9], 'b', r , true_metric[:,3], 'r') 

    plt.savefig(filename)


def save_minkowski_plot(filename, coords, minkowski, true_metric): 

    r = coords[:,1]

    plt.rcParams["figure.figsize"] = (20,20)
    plt.figure(1)
    plt.clf()
    plt.subplot(4,4,1)
    plt.title(r'$g_{tt}$', fontsize=16)
    plt.plot(r, minkowski[:,0,0], 'b', r, true_metric[:,0], 'r')

    plt.subplot(4,4,2)
    plt.title(r'$g_{tr}$', fontsize=16)
    plt.plot(r, minkowski[:,0,1], 'b')

    plt.subplot(4,4,3)
    plt.title(r'$g_{t\theta}$', fontsize=16)
    plt.plot(r, minkowski[:,0,2], 'b')

    plt.subplot(4,4,4)
    plt.title(r'$g_{t\phi}$', fontsize=16)
    plt.plot(r, minkowski[:,0,3], 'b')


    plt.subplot(4,4,5)
    plt.title(r'$g_{rt}$', fontsize=16)
    plt.plot(r, minkowski[:,1,0], 'b')

    plt.subplot(4,4,6)
    plt.title(r'$g_{rr}$', fontsize=16)
    plt.plot(r, minkowski[:,1,1], 'b', r, true_metric[:,1], 'r')

    plt.subplot(4,4,7)
    plt.title(r'$g_{r\theta}$', fontsize=16)
    plt.plot(r, minkowski[:,1,2], 'b') 

    plt.subplot(4,4,8)
    plt.title(r'$g_{r\phi}$', fontsize=16)
    plt.plot(r, minkowski[:,1,3], 'b') 


    plt.subplot(4,4,9)
    plt.title(r'$g_{\theta t}$', fontsize=16)
    plt.plot(r, minkowski[:,2,0], 'b') 

    plt.subplot(4,4,10)
    plt.title(r'$g_{\theta r}$', fontsize=16)
    plt.plot(r, minkowski[:,2,1], 'b') 

    plt.subplot(4,4,11)
    plt.title(r'$g_{\theta \theta}$', fontsize=16)
    plt.plot(r, minkowski[:,2,2], 'b', r, true_metric[:,2], 'r') 

    plt.subplot(4,4,12)
    plt.title(r'$g_{\theta\phi}$', fontsize=16)
    plt.plot(r, minkowski[:,2,3], 'b') 


    plt.subplot(4,4,13)
    plt.title(r'$g_{\phi t}$', fontsize=16)
    plt.plot(r, minkowski[:,3,0], 'b') 

    plt.subplot(4,4,14)
    plt.title(r'$g_{\phi r}$', fontsize=16)
    plt.plot(r, minkowski[:,3,1], 'b') 

    plt.subplot(4,4,15)
    plt.title(r'$g_{\phi \theta}$', fontsize=16)
    plt.plot(r, minkowski[:,3,2], 'b') 

    plt.subplot(4,4,16)
    plt.title(r'$g_{\phi \phi}$', fontsize=16)
    plt.plot(r, minkowski[:,3,3], 'b', r , true_metric[:,3], 'r') 

    plt.savefig(filename)
