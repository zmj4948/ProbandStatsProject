import numpy as np

from matplotlib import pyplot as pl

given_x_values=np.array(
    [
        -1,
        -4/5,
        -3/5,
        -2/5,
        -1/5,
        0,
        1/5,
        2/5,
        3/5,
        4/5,
        1
    ]
)

give_y_values=np.array(
    [
        -7.434582939465772,
        -6.966000985103863,
        -3.1023190231845628,
        -0.7205508129016026,
        0.3982853717151004,
        1.6164062167106008,
        3.0104305121705988,
        5.896677107660021,
        5.332269723396103,
        8.68518964579402,
        8.512125054308724
    ]
)

def find_coef_linear(points_y):
    #find coef of the best fit
    return np.polyfit(given_x_values,points_y,1)

def generate_some_error_bois(function):
    #generate y with some error based on a function
    give_me_some_y=function(given_x_values)
    error=np.random.normal(size=len(given_x_values))
    fucked_up_y=give_me_some_y+error
    return fucked_up_y

def prelimes():
    test_func=lambda x: 3*x -2

    print("Prelimiaries")
    error_bois=generate_some_error_bois(test_func)
    recovery=find_coef_linear(error_bois)
    print("Recovered coefficients "+str(recovery))
    print("First coefficient Error: "+str(3-recovery[0]))
    print("Second coefficient Error: "+str(-2-recovery[1]))


def find_coef_quad(points_y):
    #find coef of the best fit
    return np.polyfit(given_x_values,points_y,2)

def show_coeff_mean_var(index,data):
    relevant=[sublist[index] for sublist in data]
    pos="Constant"
    if index==1:
        pos="Linear"
    if index==0:
        pos="Quadratic"

    print(pos+" Coefficient:")
    print("Mean: "+str(np.mean(relevant)))
    print("Variance: "+str(np.std(relevant)))

def step_two():
    func=[
        lambda x: 3 * np.ones(len(given_x_values)),
        lambda x: 3 * x - 2,
        lambda x: 2 * x ** 2
    ]

    func_cfs = [
        [0, 0, 3],
        [0, 3, -2],
        [2, 0, 0]
    ]

    for ind, func in enumerate(func):
        cf=func_cfs[ind]
        cur_func=func
        print(str(cf[0])+"x^2+"+str(cf[1])+"*x+"+str(cf[2]))
        lots_o_coef=[]
        for _ in range(0,1000):
            error_prone_y=generate_some_error_bois(cur_func)
            recov=find_coef_quad(error_prone_y)
            lots_o_coef.append(recov)
        for coef_ind in range(0,3):
            show_coeff_mean_var(coef_ind,lots_o_coef)


def get_func_coef(func):
    lots_coef=[]
    for _ in range(0,1000):
        error_prone_y=generate_some_error_bois(func)
        recov=find_coef_quad(error_prone_y)
        lots_coef.append(recov)
    c0_list=[sublist[2] for sublist in lots_coef]
    c1_list=[sublist[1] for sublist in lots_coef]
    c2_list=[sublist[0] for sublist in lots_coef]
    return (c0_list,c1_list,c2_list)


def analyze_coef(coefs):
    co0=coefs[0]
    co1=coefs[1]
    co2=coefs[2]

    a0=co0
    a1=co1

    cov_cal=[a0,a1]

    covar_maxtrix=np.cov(cov_cal)
    covar=covar_maxtrix[0][1]

    correlation_coef=covar/np.std(a0)*np.std(a1)

    print("Means: ")
    print("c0= "+str(np.mean(a0)))
    print("c1= "+str(np.mean(a1)))

    print("Variances:")
    print(" c0 = " + str(np.std(a0) ** 2))
    print(" c1 = " + str(np.std(a1) ** 2))

    print("Covariance:")
    print(" (c0,c1) = " + str(covar))

    print("Correlation Coefficient:")
    print(" (c0,c1) = " + str(correlation_coef))

    # Create a Scatterplot
    print("\nShowing Scatterplot of coefficients...")
    pl.scatter(a0, a1)
    pl.xlabel("Constant Term")
    pl.ylabel("X Coefficient")
    pl.savefig("step_3_scatterplot.png")
    pl.show()

def analyze_coef_quad(coefs):
    c0 = coefs[0]
    c1 = coefs[1]
    c2 = coefs[2]
    print("Means:")
    print("c0 = "+str(np.mean(c0)))
    print("c1 = "+str(np.mean(c1)))
    print("c2 = "+str(np.mean(c2)))
    print("Variances:")
    print("c0 = "+str(np.std(c0)**2))
    print("c1 = "+str(np.std(c1)**2))
    print("c2 = "+str(np.std(c2)**2))
    print("Covariance:")
    print("(c0,c1) = "+str(np.cov([c0, c1])[0][1]))
    print("(c1,c2) = "+str(np.cov([c1, c2])[0][1]))
    print("(c0,c2) = "+str(np.cov([c0, c2])[0][1]))
    print("Correlation Coefficient:")
    print("(c0,c1) = "+str(np.cov([c0, c1])[0][1]/(np.std(c0)*np.std(c1))))
    print("(c1,c2) = "+str(np.cov([c1, c2])[0][1]/(np.std(c1)*np.std(c2))))
    print("(c0,c2) = "+str(np.cov([c0, c2])[0][1]/(np.std(c0)*np.std(c2))))
    print("Showing Scatterplot of coefficients...")
    pl.scatter(c0, c1)
    pl.xlabel("Linear Term")
    pl.ylabel("X Coefficient")
    pl.savefig("step_7_scatterplot_0-1.png")
    pl.show()
    pl.scatter(c1, c2)
    pl.ylabel("X^2 Coefficient")
    pl.xlabel("X Coefficient")
    pl.savefig("step_7_scatterplot_1-2.png")
    pl.show()
    pl.scatter(c0, c2)
    pl.xlabel("Linear Term")
    pl.ylabel("X^2 Coefficient")
    pl.savefig("step_7_scatterplot_0-2.png")
    pl.show()

def step_three():
    print("Analyzing coeffiences")
    print("Testin 3x-2")
    analyze_coef(get_func_coef(lambda x: 3*x-2))

def quad_step_three():
    print("Analzying coeffiences but with a Quadratic")
    pritn("Testing Function 2x^2+4x-7")
    analyze_coef_quad(get_func_coef(lambda x:2*x**2+4*x-7))