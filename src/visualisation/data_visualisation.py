import matplotlib.pyplot as plt
import matplotlib

##########################
## Global plot settings ##
##########################

font_family = "serif"
dpi = 300


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)






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
    
    plt.savefig(filename, dpi=dpi)

def save_losses_plot(filename, history):


    plt.clf()

    plt.rc('font', family=font_family)
    title_font_size = 18
    axis_font_size = 18

    samples_text="10^5"

    plt.subplot(1,3,1)
    plt.yscale("log")
    plt.title("Full Loss", fontsize=title_font_size, y=1.08)
    plt.xlabel(r"Samples ($\times " + samples_text + "$)", fontsize=axis_font_size)
    plt.ylabel("Loss", fontsize=axis_font_size)
    plt.plot(history.history['loss'])



    plt.subplot(1,3,2)
    plt.yscale("log")
    plt.title("PDE Loss", fontsize=title_font_size, y=1.08)
    plt.xlabel(r"Samples ($\times " + samples_text + "$)", fontsize=axis_font_size)
    plt.ylabel("Loss", fontsize=axis_font_size)
    plt.plot(history.history['PDE_loss'])



    plt.subplot(1,3,3)
    plt.title("BC Loss", fontsize=title_font_size, y=1.08)
    plt.xlabel(r"Samples ($\times " + samples_text + "$)", fontsize=axis_font_size)
    plt.ylabel("Loss", fontsize=axis_font_size)
    plt.yscale("log")
    plt.plot(history.history['BC_loss'])



    plt.gcf().set_size_inches(22, 6)
    plt.gcf().tight_layout()

    plt.savefig(filename, dpi=dpi)


def save_grr_plot(filename, coords, predicted_metric, true_metric): 

    r = coords[:,1] * 1e2

    plt.clf()
    plt.rc('font', family=font_family)
    title_font_size = 18
    axis_font_size = 18


#    plt.subplot(1,3,1)
#    plt.title(r'Network Prediction', fontsize=title_font_size, y=1.08)
#    plt.xlabel(r'$r$', fontsize=axis_font_size)
#    plt.ylabel(r'$g_{rr}$', fontsize=axis_font_size)
#    plt.plot(r, predicted_metric[:,1,1], 'b')
#
#
#    plt.subplot(1,3,2)
#    plt.title(r'Analytical Solution', fontsize=title_font_size, y=1.08)
#    plt.xlabel(r'$r$', fontsize=axis_font_size)
#    plt.ylabel(r'$g_{rr}$', fontsize=axis_font_size)
#    plt.plot(r, true_metric[:,1,1], 'r')
#
#
#    plt.subplot(1,3,3)
    plt.title(r'PINN $g_{rr}$ prediction', fontsize=title_font_size, y=1.08)
#    plt.yscale('log')
    plt.xlabel(r'$r$ (m)', fontsize=axis_font_size)
    plt.ylabel(r'$g_{rr}$', fontsize=axis_font_size)
    plt.plot(r, predicted_metric[:,1,1], 'b', label="Calculated From the Analytical Metric")
    plt.plot(r, true_metric[:,1,1], 'r', label="Analytical Solution")

    plt.legend(loc='center left', bbox_to_anchor=(1.5,0.5), prop={'size': 18})
    
    plt.gcf().set_size_inches(18, 7)
    plt.gcf().tight_layout()

    plt.savefig(filename, dpi=dpi)


def save_4_4_tensor(filename, tensor_name,  coords, tensor, description, suptitle, true_G): 

    r = coords[:,1] * 1e2

    plt.clf()
    plt.rc('font', family=font_family)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    title_font_size = 22
    axis_font_size = 22
    plt.suptitle(suptitle, fontsize=25)

    coord_names = [r"t ", r"r ", r"\theta ", r"\phi "]

    for i, coord_name_1 in enumerate(coord_names):
        for j, coord_name_2 in enumerate(coord_names):

            plt.subplot(4,4, 4*i + j + 1)
#            plt.title(description + r' $' + tensor_name + '_{' + coord_name_1 + coord_name_2 +  r'}$', fontsize=title_font_size, y=1.08)
            plt.xlabel(r'$r$ (m)', fontsize=axis_font_size)
            plt.ylabel(r' $' + tensor_name + r'^{' + coord_name_1 + coord_name_2 +  r'}$ ($m^{-2}$)', fontsize=axis_font_size)

            plt.plot(r, tensor[:,i,j], 'b', label="PINN prediction")
            plt.plot(r, true_G[:,i,j], 'r', label="Analytical Solution")

            plt.xticks([5,7.5,10])

    plt.gcf().set_size_inches(20, 20)
    plt.gcf().tight_layout(pad=2.0)

    plt.savefig(filename+".jpg", dpi=dpi)

