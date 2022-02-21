import matplotlib.pyplot as plt



def save_fixed_point_plots(filename, history):


    fixed_point_results = {key: value for key, value in history.history.items() if key not in { "loss", "PDE_loss", "BC_loss"} }

    num_fp = len(fixed_point_results)

    
    plt.figure(1, (6.5 * num_fp,6))
    plt.clf()


    for i, (radius, fp_values) in enumerate(fixed_point_results.items()):

        plt.subplot(1,num_fp + 1, i+1)
        plt.title(f"r = {radius}")
        plt.xlabel("Samples (x1000)")
        plt.ylabel(r"$g_tr$")
        plt.plot(fp_values)


    plt.gcf().set_size_inches(6.1*num_fp, 6)
    plt.gcf().tight_layout()
    
    plt.savefig(filename)

def save_losses_plot(filename, history):


    plt.clf()

    plt.subplot(1,3,1)
    plt.yscale("log")
    plt.title("Full Loss")
    plt.xlabel("Samples (x1000)")
    plt.ylabel("Loss")
    plt.plot(history.history['loss'])


    plt.subplot(1,3,2)
    plt.yscale("log")
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



    plt.gcf().set_size_inches(22, 6)
    plt.gcf().tight_layout()

    plt.savefig(filename)


def save_grr_plot(filename, coords, predicted_metric, true_metric): 

    r = coords[:,1]

    plt.clf()
    
    plt.subplot(1,3,1)
    plt.title(r'$g_{rr} predicted$')
    plt.xlabel(r'$r$')
    plt.plot(r, predicted_metric[:,1,1], 'b')


    plt.subplot(1,3,2)
    plt.title(r'$g_{rr} actual$')
    plt.xlabel(r'$r$')
    plt.plot(r, true_metric[:,1,1], 'r')


    plt.subplot(1,3,3)
    plt.title(r'$g_{rr} both$')
    plt.yscale('log')
    plt.xlabel(r'$r$')
    plt.plot(r, predicted_metric[:,1,1], 'b', r, true_metric[:,1,1], 'r')
    
    plt.gcf().set_size_inches(20, 6)
    plt.gcf().tight_layout()

    plt.savefig(filename)


def save_4_4_tensor(filename, tensor_name,  coords, tensor): 

    r = coords[:,1]

    plt.clf()
   
    coord_names = [r"t ", r"r ", r"\theta ", r"\phi "]

    for i, coord_name_1 in enumerate(coord_names):
        for j, coord_name_2 in enumerate(coord_names):

            plt.subplot(4,4, 4*i + j + 1)
            plt.title(r'$' + tensor_name + '_{' + coord_name_1 + coord_name_2 +  r'} predicted$')
            plt.xlabel(r'$r$')
            plt.plot(r, tensor[:,i,j], 'b')

    plt.gcf().set_size_inches(20, 20)
    plt.gcf().tight_layout()

    plt.savefig(filename+".jpg")

