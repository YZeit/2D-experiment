import streamlit as st
from docplex.mp.sdetails import SolveDetails
import matplotlib.pyplot as plt
import numpy as np


def app():
    st.header('Bi-objective experiment')
    #example bi-objective linear program
    st.subheader('Example:')

    st.latex(r'''
        \begin{array}{rll}
         \min & f_1(x) = 3x1 + x2 & \\
         & f_2(x) = -x1 + 4x2 & \\
         s.t.: & x_2 \leqslant 6 & \\
         & x1 + 3x2 \geqslant 3 & \\
         & 2x1 - x2 \leqslant 6 & \\
         & 2x1 + x2 \leqslant 10 & \\
         & x1, x2 \geqslant 0  & \\
         & x1, x2 \in \mathbb{Z}  & \\
    \end{array}
    ''')

    reference_point = [0,0]
    st.sidebar.header('Reference Point')
    reference_point[0] = st.sidebar.slider("Please select the reference point for the 1st objective", min_value=-10.00, max_value=25.00, value=0.00)
    reference_point[1] = st.sidebar.slider("Please select the reference point for the 2nd objective", min_value=-10.00, max_value=25.00, value=0.00)


    weighting_normalized = [0.5,0.5]
    #weighting_normalized[0] = st.sidebar.slider('Please select the weighting for the 1st objective', min_value=0.0, max_value=1.0, value=0.5)
    #weighting_normalized[1] = st.sidebar.slider('Please select the weighting for the 2nd objective', min_value=0.0, max_value=1.0, value=1-weighting_normalized[0])

    #a_f1 = 0.6
    #a_f2 = 0.3
    #a_f1f2 = 0.1

    f1_nadir = 14
    f2_nadir = 24

    f1_ideal = 1
    f2_ideal = -3

    # Optimization Model
    from docplex.mp.model import Model
    m = Model(name='Bi-objective convex Pareto Front')

    roh = 0.000001  # arbitrary positive small value

    # decision variables
    x1 = m.integer_var(lb=0)
    x2 = m.integer_var(lb=0)
    nu = m.continuous_var(name='nu')

    # constraints
    constraint1 = x2 <= 6
    constraint2 = x1 + 3 * x2 >= 3
    constraint3 = 2 * x1 - x2 <= 6
    constraint4 = 2 * x1 + x2 <= 10
    m.add_constraint(constraint1)
    m.add_constraint(constraint2)
    m.add_constraint(constraint3)
    m.add_constraint(constraint4)

    # objective functions
    objective = [0, 0]
    objective[0] = 3 * x1 + x2
    objective[1] = -x1 + 4 * x2

    feasible_solutions_z1 = []
    feasible_solutions_z2 = []

    for i in range(1,15):
        m.add_constraint(objective[0] == i, ctname='temp1')
        for i in range(-3, 25):
            m.add_constraint(objective[1] == i, ctname='temp2')
            try:
                m.minimize(objective[0])
                m.solve()
                # st.write('iteration: ' + str(i))
                # st.write(objective[0].solution_value)
                feasible_solutions_z1.append(objective[0].solution_value)
                # st.write(objective[1].solution_value)
                feasible_solutions_z2.append(objective[1].solution_value)
                m.remove_constraint('temp2')
            except:
                m.remove_constraint('temp2')
        m.remove_constraint('temp1')




    #st.write(feasible_solutions_z1)
    #st.write(feasible_solutions_z2)


    st.set_option('deprecation.showPyplotGlobalUse', False)

    #st.header('The feasible region')
    #extreme points x1 and x2
    #extreme_points_x1 = [0, 0, 3, 4, 2, 0]
    #extreme_points_x2 = [6, 1, 0, 2, 6, 6]
    #plt.plot(extreme_points_x1, extreme_points_x2)
    #plt.xlabel("$x_1$")
    #plt.ylabel("$x_2$")
    #st.pyplot()

    #Choquet
    st.header('Choquet Integral')
    st.latex(r'''C_\mu(x)=\sum_{\{i\}\subseteq G} a(\{i\})(z_i(x)) + \sum_{\{i,j\}\subseteq G} a(\{i,j\})\min(z_i(x),z_j(x))''')
    st.latex(r'''{\displaystyle \Delta(x)=\frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )},\ldots,\frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )},\ldots,\frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad})}}''')
    st.latex(r'''\min_{x \in X} C_\mu \big(\Delta(x)\big)''')
    st.latex(r'''\begin{array}{rcl}
    {\displaystyle C_\mu(\Delta x)}  & = & {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\})\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right) \; +} \\  
    & + &  {\displaystyle\sum_{\{i,j\}\subseteq G} a(\{i,j\})\min\left\{\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right),\; \left(\frac{z_j^r-z_j (x)}{z_j^r-z_j^{nad}}\right)\right\}},
\end{array} ''')
    st.write('''To linearize the problem, let's introduce a new variable $u$:''')
    st.latex(r'''{\displaystyle u=\min \big(\frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )},\ldots,\frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )},\ldots,\frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad})}}\big)''')
    st.latex(r'''\begin{array}{l}
    {\displaystyle u \geqslant \frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )} - My_1} \\
    {\displaystyle ...} \\
    {\displaystyle u \geqslant \frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )} - My_k} \\
        ... \\
    {\displaystyle u \geqslant \frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad} )} - My_p} \\
    {\displaystyle \sum_{k=1}^p y_k = p-1}
    \end{array}''')
    st.write('To ensure that the minimization model also works for a negative value of $a(\{i,j\})$ (e.g., in case of overlap effect between $i$ and $j$) following constraints are introduced:')
    st.latex(r'''\begin{array}{l}
    {\displaystyle u \leqslant \frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )}} \\
    {\displaystyle ...} \\
    {\displaystyle u \leqslant \frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )}} \\
        ... \\
    {\displaystyle u \leqslant \frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad} )}}
    \end{array}''')
    st.subheader('New proposal: Choquet min-norm')
    st.write('Please have a look at our overleaf section 7')
    st.subheader('Preference Assumptions:')
    st.write('Please select the preference assumptions in the sidebar')
    st.sidebar.header('Preference Information')
    st.sidebar.write('$a(\{z_1\})$')
    a_f1 = st.sidebar.slider('Please select the value for the 1st criterion', min_value=0.00, max_value=1.00,
                     value=0.60)
    st.sidebar.write('$a(\{z_2\})$')
    a_f2 = st.sidebar.slider('Please select the value for the 2nd criterion', min_value=0.00, max_value=1.00,
                     value=0.30)
    st.sidebar.write(
        '$a(\{z_1,z_2\})$ (_Please note: this value will be automatically updated with regard to the condition shown below_)')
    a_f1f2 = round(1 - a_f1 - a_f2, 2)
    st.sidebar.write(str(a_f1f2))
    st.write('With regard to the 2-additive measures, following formulations must be ensured:')
    st.latex(r'''\begin{array}{rl}
        {\displaystyle a(\emptyset)=0,}  &  {\displaystyle \sum_{i \in G} a(\{i\}) + \sum_{\{i,j\} \subseteq G} a(\{i,j\}) = 1} \\ 
        {\displaystyle a(\{i\}) \geqslant 0,\forall i \in G,} & {\displaystyle  a(\{i\}) + \sum_{j \in T} a(\{i,j\}) \geqslant 0, \forall i \in G, \forall T \subseteq G \setminus \{i\}.}
    \end{array} ''')

    from docplex.mp.model import Model
    m1 = Model(name='Optimization Model')


    roh = 0.000001  # arbitrary positive small value

    # decision variables
    x1 = m1.integer_var(lb=0)
    x2 = m1.integer_var(lb=0)
    nu = m1.continuous_var(name='nu')

    # constraints
    constraint1 = x2 <= 6
    constraint2 = x1 + 3 * x2 >= 3
    constraint3 = 2 * x1 - x2 <= 6
    constraint4 = 2 * x1 + x2 <= 10
    m1.add_constraint(constraint1)
    m1.add_constraint(constraint2)
    m1.add_constraint(constraint3)
    m1.add_constraint(constraint4)

    # objective functions
    objective = [0, 0]
    objective[0] = 3 * x1 + x2
    objective[1] = -x1 + 4 * x2

    bigm = 100000000
    y_1 = m1.binary_var()
    y_2 = m1.binary_var()
    #min_value = m.continuous_var()
    nu_1 = m1.continuous_var()
    nu_2 = m1.continuous_var()
    nu_3 = m1.continuous_var()
    delta_x1 = m1.continuous_var()
    delta_x2 = m1.continuous_var()


    #m.add_constraint(min_value <= delta_x1+(bigm*y_1))
    #m.add_constraint(min_value <= delta_x2+(bigm*y_2))
    #m.add_constraint(y_1+y_2==1)
    #m_lambda = Model(name='lambda')
    #lambda_1 = m_lambda.continuous_var()
    #m_lambda.add_constraint(f1_nadir-lambda_1*(f1_nadir-reference_point[0])<=f1_ideal)
    #m_lambda.add_constraint(f2_nadir - lambda_1 * (f2_nadir - reference_point[1]) <= f2_ideal)
    #m_lambda.minimize(lambda_1)
    #m_lambda.solve()
    #z1_ref_star = f1_nadir-lambda_1.solution_value*(f1_nadir-reference_point[0])
    #z2_ref_star = f2_nadir - lambda_1.solution_value * (f2_nadir - reference_point[1])
    #st.write(z1_ref_star)
    #st.write(z2_ref_star)
    z1_ref_star = reference_point[0]
    z2_ref_star = reference_point[1]

    lambda_1_1 = a_f1
    lambda_1_2 = (1- a_f1)
    lambda_2_1 = (1-a_f2)
    lambda_2_2 = a_f2

    m1.add_constraint(delta_x1 == ((objective[0]-z1_ref_star)/(f1_nadir-z1_ref_star)))
    m1.add_constraint(delta_x2 == ((objective[1]-z2_ref_star)/(f2_nadir-z2_ref_star)))
    m1.add_constraint(delta_x1 >= 0)
    m1.add_constraint(delta_x2 >= 0)
    choquet_1 = (lambda_1_1*delta_x1+lambda_1_2*delta_x2)
    choquet_2 = (lambda_2_1*delta_x1+lambda_2_2*delta_x2)
    if a_f1f2>=0:
        m1.add_constraint(nu_1 >= choquet_1-(bigm*y_1))
        m1.add_constraint(nu_1 >= choquet_2-(bigm*y_2))
    else:
        m1.add_constraint(nu_1 >= choquet_1)
        m1.add_constraint(nu_1 >= choquet_2)

    choquet_11 = lambda_1_1*delta_x1
    choquet_12 = lambda_1_2*delta_x2
    choquet_21 = lambda_2_1*delta_x1
    choquet_22 = lambda_2_2*delta_x2

    #m.add_constraint(nu_1 >= lambda_1_1*delta_x1+lambda_1_2*delta_x2-(bigm*y_1))
    #m.add_constraint(nu_1 >= lambda_2_1*delta_x1+lambda_2_2*delta_x2-(bigm*y_2))
    #m.add_constraint(nu_1 >= choquet_11)
    #m.add_constraint(nu_1 >= choquet_12)
    #m.add_constraint(nu_2 >= choquet_21)
    #m.add_constraint(nu_2 >= choquet_22)

    #m.add_constraint((nu_3 >= nu_1))
    #m.add_constraint((nu_3 >= nu_2))
    #m.add_constraint((nu_3 <= nu_1+(bigm*y_1)))
    #m.add_constraint((nu_3 <= nu_2+(bigm*y_2)))

    #m.add_constraint((nu_3 >= nu_1-(bigm*y_1)))
    #m.add_constraint((nu_3 >= nu_2-(bigm*y_2)))

    #m.add_constraint(min_value >= delta_x1-(bigm*y_1))
    #m.add_constraint(min_value >= delta_x2-(bigm*y_2))
    #m.add_constraint(min_value <= delta_x1 + y_1*bigm)
    #m.add_constraint(min_value <= delta_x2 + y_2*bigm)

    #m.add_constraint(min_value <= delta_x1+(bigm*y_1))
    #m.add_constraint(min_value <= delta_x2+(bigm*y_2))
    #m.add_constraint(min_value*a_f1f2>=0)
    m1.add_constraint(y_1+y_2==1)

    #m.add_constraint(a_f1 * delta_x1 + (a_f1f2 * min_value) <= nu_1)
    #m.add_constraint(a_f1 * delta_x1 + (a_f1f2 * min_value) <= nu_1)
    #m.add_constraint(a_f2 * delta_x2 + (a_f1f2 * min_value) <= nu_1)
    #m.add_constraint(a_f1 * delta_x1 + (a_f1f2 * delta_x2) <= nu_1)
    #m.add_constraint(a_f2 * delta_x2 + (a_f1f2 * delta_x1) <= nu_1)
    #m.add_constraint(delta_x1-delta_x2<=bigm*(1-y))
    #m.add_constraint(delta_x2-delta_x1<=bigm*y)
    #m.add_constraint(delta_x1 >= min_value)
    #m.add_constraint(delta_x2 >= min_value)
    #m.add_constraint(delta_x1 - bigm*(1-y) <= min_value)
    #m.add_constraint(delta_x2 - bigm*y <= min_value)
    #delta_x1_plus = m.continuous_var(lb=0)
    #delta_x1_minus = m.continuous_var(lb=0)
    #delta_x2_plus = m.continuous_var(lb=0)
    #delta_x2_minus = m.continuous_var(lb=0)
    #m.add_constraint(delta_x1==delta_x1_plus-delta_x1_minus)
    #m.add_constraint(delta_x2 == delta_x2_plus-delta_x2_minus)
    #m.add_constraint(delta_x1_plus-delta_x1_minus-delta_x2_plus+delta_x2_minus <= bigm*(1-y))
    #m.add_constraint(delta_x2_plus-delta_x2_minus-delta_x1_plus+delta_x1_minus <= bigm*y)
    #m.add_constraint(delta_x1_plus-delta_x1_minus >= min_value)
    #m.add_constraint(delta_x2_plus-delta_x2_minus >= min_value)
    #m.add_constraint(delta_x1_plus-delta_x1_minus - bigm*(1-y) <= min_value)
    #m.add_constraint(delta_x2_plus-delta_x2_minus - bigm*y <= min_value)
    #alpha_1 = m.binary_var()
    #alpha_2 = m.binary_var()
    #alpha = m.binary_var()
    #m.add_constraint(delta_x1_minus-delta_x1_plus <= bigm*alpha_1)
    #m.add_constraint(delta_x1_plus-delta_x1_minus <= bigm*(1-alpha_1))
    #m.add_constraint(delta_x2_minus-delta_x2_plus <= bigm*alpha_2)
    #m.add_constraint(delta_x2_plus-delta_x2_minus <= bigm*(1-alpha_2))
    #m.add_constraint(alpha_1 <= alpha)
    #m.add_constraint(alpha_2 <= alpha)

    #m.add_constraint(min_value == min(delta_x1,delta_x2))
    #m.add_constraint(objective[0]<=20)
    #choquet = a_f1 * delta_x1 + a_f2*delta_x2 + (a_f1f2 * min_value)
    #m.add_constraint(a_f1*((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir)) + a_f2*((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir)) + a_f1f2*min_value <= nu)
    #m.add_constraint(a_f1*((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir)) <= nu)
    #m.add_constraint(a_f2*((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir)) <= nu)
    m1.minimize(nu_1)
    m1.solve()
    solve_details = m1.solve_details
    #st.write(solve_details)


    x1_solution_choquet = x1.solution_value
    x2_solution_choquet = x2.solution_value
    z1_solution_choquet = objective[0].solution_value
    z2_solution_choquet = objective[1].solution_value

    min_value_scatter_x = min(-10,round(z1_ref_star))
    min_value_scatter_y = min(-10, round(z2_ref_star))


    delta_x1 = ((objective[0].solution_value-z1_ref_star)/(f1_nadir-z1_ref_star))
    delta_x2 = ((objective[1].solution_value-z2_ref_star)/(f2_nadir-z2_ref_star))
    choquet = a_f1*delta_x1 + a_f2*delta_x2 + min(delta_x1, delta_x2)*a_f1f2
    choquet_1 = lambda_1_1*delta_x1+lambda_1_2*delta_x2
    choquet_2 = lambda_2_1*delta_x1+lambda_2_2*delta_x2

    st.subheader('Solution')
    st.write('$x_1$: ' + str(x1_solution_choquet))
    st.write('$x_2$: ' + str(x2_solution_choquet))
    st.write('$z_1(x)$: ' + str(z1_solution_choquet))
    st.write('$z_2(x)$: ' + str(z2_solution_choquet))
    #st.write('min value: ' + str(min_value.solution_value))
    st.write('delta x_1: ' + str(delta_x1))
    st.write('delta x_2: ' + str(delta_x2))
    st.write('choquet: ' + str(choquet))
    st.write('nu_1:' + str(nu_1.solution_value))
    st.write('nu_2:' + str(nu_2.solution_value))
    st.write('nu_3:' + str(nu_3.solution_value))
    st.write('y_1:' + str(y_1.solution_value))
    st.write('y_2:' + str(y_2.solution_value))
    st.write('choquet_11:' + str(choquet_11.solution_value))
    st.write('choquet_12:' + str(choquet_12.solution_value))
    st.write('choquet_21:' + str(choquet_21.solution_value))
    st.write('choquet_22:' + str(choquet_22.solution_value))

    st.write('choquet_1:' + str(choquet_11.solution_value+choquet_12.solution_value))
    st.write('choquet_2:' + str(choquet_21.solution_value+choquet_22.solution_value))

    st.header('The projected feasible region')
    fig2, ax1 = plt.subplots()
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.xticks(range(min_value_scatter_x,26))
    plt.yticks(range(min_value_scatter_y, 26))
    #plt.yticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    #            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    #plt.xticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    #            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    plt.grid(True)
    #ax1 = fig2.add_subplot()
    ax1.set_xlim([min_value_scatter_x, 25])
    ax1.set_ylim([min_value_scatter_y, 25])
    ax1.tick_params(axis='both', which='major', labelsize=5)
    ax1.tick_params(axis='both', which='minor', labelsize=4)
    extreme_points_z1 = [6, 1, 9, 14, 12, 6]
    extreme_points_z2 = [24, 4,-3, 4, 22, 24]
    #optimizing_factor = st.slider('Please optimize until the metric reach the feasible region', min_value=0.00, max_value=1.00, value=1.00)
    optimizing_factor = choquet
    schnittpunkt_z1 = z1_ref_star + optimizing_factor * (f1_nadir-z1_ref_star)
    schnittpunkt_z2 = z2_ref_star + optimizing_factor * (f2_nadir-z2_ref_star)
    #schnittpunkt_z1 = z1_ref_star + optimizing_factor
    #schnittpunkt_z2 = z2_ref_star + optimizing_factor

    z_1_1 = np.arange(0, schnittpunkt_z1, 0.1)
    z_1_2 = np.arange(schnittpunkt_z1, 1/a_f1, 0.1)
    z_2_1 = (1/a_f2 - (((a_f1+a_f1f2)/a_f2) * z_1_1))
    z_2_2 = (1/(a_f2+a_f1f2) - ((a_f1/(a_f2+a_f1f2)) * z_1_2))

    z1_choquet_1 = z1_ref_star
    z2_choquet_1 = z2_ref_star + optimizing_factor * 1/(a_f2/(f2_nadir-z2_ref_star))
    z1_choquet_2 = z1_ref_star + optimizing_factor * 1/(a_f1/(f1_nadir-z1_ref_star))
    z2_choquet_2 = z2_ref_star

    z111_choquet_tschebycheff = z1_ref_star - optimizing_factor * 1 / (lambda_1_1/(f1_nadir-z1_ref_star))
    z121_choquet_tschebycheff = z1_ref_star + optimizing_factor * 1 / (lambda_1_1 / (f1_nadir - z1_ref_star))
    z211_choquet_tschebycheff = z2_ref_star - optimizing_factor * 1 / (lambda_1_2/(f2_nadir-z2_ref_star))
    z221_choquet_tschebycheff = z2_ref_star + optimizing_factor * 1 / (lambda_1_2 / (f2_nadir - z2_ref_star))

    z112_choquet_tschebycheff = z1_ref_star - optimizing_factor * 1 / (lambda_2_1/(f1_nadir-z1_ref_star))
    z122_choquet_tschebycheff = z1_ref_star + optimizing_factor * 1 / (lambda_2_1 / (f1_nadir - z1_ref_star))
    z212_choquet_tschebycheff = z2_ref_star - optimizing_factor * 1 / (lambda_2_2/(f2_nadir-z2_ref_star))
    z222_choquet_tschebycheff = z2_ref_star + optimizing_factor * 1 / (lambda_2_2 / (f2_nadir - z2_ref_star))


    ax1.plot(extreme_points_z1, extreme_points_z2, dashes=[6, 2], linewidth=1, label='feasible area')
    ax1.plot([z1_ref_star, z1_choquet_1, schnittpunkt_z1, z1_choquet_2, z1_ref_star], [z2_ref_star, z2_choquet_1, schnittpunkt_z2, z2_choquet_2, z2_ref_star], dashes=[6, 2], c='orange', linewidth=1, label='Choquet metric')
    #ax1.plot([z1_choquet_1, z1_choquet_2],
     #        [z2_choquet_1, z2_choquet_2], dashes=[6, 2], c='yellow',
      #       linewidth=1, label='Choquet metric')
    #ax1.plot(z_1_2+reference_point[0], z_2_2+reference_point[1], dashes=[6, 2], c='orange', linewidth=1)
    #ax1.plot([reference_point[0], reference_point[0]], [reference_point[1], 1/a_f2 + reference_point[1]], dashes=[6, 2], c='orange', linewidth=1)
    #ax1.plot([reference_point[0], 1 / a_f1 + reference_point[0]], [reference_point[1], reference_point[1]],
    #         dashes=[6, 2], c='orange', linewidth=1)
    #ax1.plot([z111_choquet_tschebycheff, z121_choquet_tschebycheff, z121_choquet_tschebycheff, z111_choquet_tschebycheff, z111_choquet_tschebycheff],
    #         [z221_choquet_tschebycheff, z221_choquet_tschebycheff, z211_choquet_tschebycheff, z211_choquet_tschebycheff, z221_choquet_tschebycheff], dashes=[6, 2], c='orange',
    #         linewidth=1, label='Choquet metric')
    #ax1.plot([z112_choquet_tschebycheff, z122_choquet_tschebycheff, z122_choquet_tschebycheff, z112_choquet_tschebycheff, z112_choquet_tschebycheff],
    #         [z222_choquet_tschebycheff, z222_choquet_tschebycheff, z212_choquet_tschebycheff, z212_choquet_tschebycheff, z222_choquet_tschebycheff], dashes=[6, 2], c='orange',
    #         linewidth=1, label='Choquet metric')
    ax1.scatter(feasible_solutions_z1, feasible_solutions_z2, s=6, c='b', label='feasible solutions')
    ax1.scatter(z1_choquet_1, z2_choquet_1, s=6, c='gray')
    ax1.scatter(z1_choquet_2, z2_choquet_2, s=6, c='gray')
    ax1.scatter(schnittpunkt_z1, schnittpunkt_z2, s=6, c='gray')
    #ax1.scatter(schnittpunkt_z1+reference_point[0], schnittpunkt_z2+reference_point[1], s=6, c='b', label='feasible solutions')
    ax1.scatter(reference_point[0], reference_point[1], color='y', label='reference point ($z^{r}$)')
    ax1.scatter(z1_solution_choquet, z2_solution_choquet, color='g', label='obtained solution')
    ax1.scatter(f1_ideal, f2_ideal, c='lightgreen', label='ideal point')
    ax1.scatter(f1_nadir, f2_nadir, c='red', label='nadir point')
    ax1.plot([z1_ref_star, f1_nadir], [z2_ref_star, f2_nadir], color='black', linewidth=1, dashes=[6,2])
    #ax1.scatter(z1_ref_star, z2_ref_star, label='transformed reference point ($z^{r*}$)')
    ax1.plot([z1_ref_star, z1_solution_choquet], [z2_ref_star, z2_solution_choquet])
    ax1.legend(fontsize='x-small')
    st.pyplot(fig2)



