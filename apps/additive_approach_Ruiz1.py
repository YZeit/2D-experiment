import streamlit as st
from docplex.mp.sdetails import SolveDetails
import matplotlib.pyplot as plt
import numpy as np


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
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

    # Reference point position
    reference_point = [0,0]
    st.sidebar.header('Reference Point')
    reference_point[0] = st.sidebar.slider("Please select the reference point for the 1st objective", min_value=-10.00, max_value=25.00, value=0.00)
    reference_point[1] = st.sidebar.slider("Please select the reference point for the 2nd objective", min_value=-10.00, max_value=25.00, value=0.00)

    # nadir and ideal solution
    f1_nadir = 14
    f2_nadir = 24
    f1_ideal = 1
    f2_ideal = -3

    # Optimization Model
    from docplex.mp.model import Model
    m = Model(name='Bi-objective convex Pareto Front')

    # decision variables
    x1 = m.integer_var(lb=0)
    x2 = m.integer_var(lb=0)

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
    objective[1] = 4 * x2 - x1

    # compute the feasible region
    feasible_solutions_z1 = []
    feasible_solutions_z2 = []
    for i in range(1,15):
        m.add_constraint(objective[0] == i, ctname='temp1')
        for j in range(-4, 25):
            m.add_constraint(objective[1] == j, ctname='temp2')
            try:
                m.minimize(objective[0])
                m.solve()
                #st.write(f'solution for z1 {i} and z2 {j}: {objective[0].solution_value} and {objective[1].solution_value}')
                # st.write('iteration: ' + str(i))
                # st.write(objective[0].solution_value)
                feasible_solutions_z1.append(objective[0].solution_value)
                # st.write(objective[1].solution_value)
                feasible_solutions_z2.append(objective[1].solution_value)
                m.remove_constraint('temp2')
            except:
                m.remove_constraint('temp2')
        m.remove_constraint('temp1')

    #Choquet description
    st.header('Choquet Integral')
    st.latex(r'''C_\mu(x)=\sum_{\{i\}\subseteq G} a(\{i\})(z_i(x)) + \sum_{\{i,j\}\subseteq G} a(\{i,j\})\min(z_i(x),z_j(x))''')
    st.latex(r'''{\displaystyle \Delta(x)=\frac{\big(z_1^r-z_1 (x)\big)}{(z_1^r-z_1^{nad} )},\ldots,\frac{\big(z_k^r-z_k (x)\big)}{(z_k^r-z_k^{nad} )},\ldots,\frac{\big(z_p^r-z_p (x)\big)}{(z_p^r-z_p^{nad})}}''')
    st.latex(r'''\min_{x \in X} C_\mu \big(\Delta(x)\big)''')
    st.latex(r'''\begin{array}{rcl}
    {\displaystyle C_\mu(\Delta x)}  & = & {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\})\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right) \; +} \\  
    & + &  {\displaystyle\sum_{\{i,j\}\subseteq G} a(\{i,j\})\min\left\{\left(\frac{z_i^r-z_i (x)}{z_i^r-z_i^{nad}}\right),\; \left(\frac{z_j^r-z_j (x)}{z_j^r-z_j^{nad}}\right)\right\}},
\end{array} ''')
    
    st.subheader('The Choquet program')
    st.write('To find the feasible solution $x$ that is closest to the reference point with respect to the introduced Choquet distance, we look for a feasible solution $x$ minimizing $C_\mu (\Delta (x))$. The following optimization program can be defined as the Choquet program:')
    st.latex(r'''\min_{x \in X} C_\mu (\Delta(x))''')
    st.write('Through replacing $C_\mu(\Delta(x))$ by the Choquet metric with respect to the 2-additive fuzzy measure, the Choquet program can be written as:')
    st.latex(r'''\min_{x \in X} \left\{ \sum_{\{i\}\subseteq G} a(\{i\})\Delta_i (x) +\sum_{\{i,j\}\subseteq G} a(\{i,j\})\min\left\{\Delta_i (x), \Delta_j (x)\right\} \right\}''')
    st.write(' In order to linearize the $\min$ formulation in the previous equation, we introduce a new non-negative variable $ \beta $. The linearized Choquet program can then be formulated as:')
    st.latex(r'''\begin{array}{rl}
         \min               &  {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\}) \Delta_i (x) + \sum_{\{i,j\}\subseteq G} a(\{i,j\}) \beta(\{i,j\})}  \\
         subject to: & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_i (x) - My_{(\{i,j\}),1}}, \;\; \forall (\{i,j\}) \subseteq G\\
         & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_j (x) - My_{(\{i,j\}),2}}, \;\; \forall (\{i,j\}) \subseteq G\\
         & {\displaystyle y_{(\{i,j\}),1} + y_{(\{i,j\}),2} = 1}, \;\; \forall (\{i,j\}) \subseteq G \\
                            & x \in X \\ 
                            & \beta(\{i,j\}) \geqslant 0,\;\; \forall (\{i,j\}) \subseteq G . 
    \end{array}''')
    st.write('with $M$ being an arbitrary big value.')
    st.write('''Using the ideal outcome vector as reference point ensures to find a supported efficient solution. However, the proposed Choquet metric in its current setting has two main drawbacks that result in the program's inability to find all non-dominated solutions:''')
    st.write('1. negative distances to the reference point can not be considered')
    st.write('2. solutions that dominate the reference point can not be identified')
    st.write('Among these issues, the Choquet metric is limited in reaching unsupported non-dominated solutions as it is also the case for the simple additive metric (Ruiz et al, 2008). To overcome these issues, we will present several adjustments to the initially proposed Choquet program.')         
    st.write('''The first problem results from the Choquet integral's requirements that do not allow aggregating negative values. Given the example shown in overleaf one could argue that solution C should be preferable over solution A, since it significantly improves the $z_2$ component while the distance to the $z_1$ component is smaller than the $z_2$ distance of solution A to the reference point. However, the Choquet program is not able to observe solution C with the selected reference point.''')
    st.write('Based on the approach to avoid penalizing negative deviations for an additive achievement scalarizing function shown in (Ruiz et al, 2008), a modification of the Choquet program that only allows positive deviations to be considered is introduced:')
    st.latex(r'''    \begin{array}{rl}
         \min &  {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\}) \max \{\Delta_i (x), 0\} + \sum_{\{i,j\}\subseteq G} a(\{i,j\}) \max \{\beta(\{i,j\}), 0\}}  \\
         subject to: & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_i (x) - My_{(\{i,j\}),1}}, \;\; \forall (\{i,j\}) \subseteq G\\
         & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_j (x) - My_{(\{i,j\}),2}}, \;\; \forall (\{i,j\}) \subseteq G\\
         & {\displaystyle y_{(\{i,j\}),1} + y_{(\{i,j\}),2} = 1}, \;\; \forall (\{i,j\}) \subseteq G \\
        & x \in X.
    \end{array}''')
    st.write('To make the modified program differentiable, the model shown above is linearized, introducing new binary variable $v$ and $w$:')
    st.latex(r'''  \begin{array}{rl}
         \min &  {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\}) \Delta_i (x) v_i + \sum_{\{i,j\}\subseteq G} a(\{i,j\})  \beta(\{i,j\}) w_{i,j}}  \\
         subject to: & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_i (x) - My_{(\{i,j\}),1}}, \;\; \forall (\{i,j\}) \subseteq G\\
        & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_j (x) - My_{(\{i,j\}),2}}, \;\; \forall (\{i,j\}) \subseteq G\\
        & {\displaystyle y_{(\{i,j\}),1} + y_{(\{i,j\}),2} = 1}, \;\; \forall (\{i,j\}) \subseteq G \\
        & \Delta_i(x) \leqslant Mv_i, \forall (\{i\}) \subseteq G \\
        & \Delta_i(x) \geqslant M(v_i-1), \forall (\{i\}) \subseteq G \\
        & \beta(\{i,j\}) \leqslant Mw_{i,j}, \forall (\{i,j\}) \subseteq G \\
        & \beta(\{i,j\}) \geqslant M(w_{i,j}-1), \forall (\{i,j\}) \subseteq G \\
        & v_i \in {0,1}, \forall (\{i\}) \subseteq G \\
        & w_{i,j} \in {0,1}, \forall (\{i,j\}) \subseteq G \\
        & x \in X. 
    \end{array}   ''')
    st.write('As it can be seen in overleaf the modified Choquet program is able to reach solution C which is preferable over solution A due to the aforementioned arguments. Negative deviations are no longer penalized, through which the program can identify solutions with negative distance components to the reference point whenever there exists at least one distance component having a positive value.')         
    st.write('However, if there is a solution that dominates the reference point, which means that all distance components are negative, the proposed Choquet program is not able to find that solution. As the same issue already appeared for the simple additive $L_1$-norm, Ruiz et al (2008) proposed a procedure in which another program first verifies whether there exists a solution that dominates the selected reference point or not. In case there exists a solution that dominates the reference point, Ruiz et al (2008) proposed to simply minimize the weighted additive distance of the problem at hand. Since we want to additionally consider interaction between criteria, further adjustments are required. A possible way to find non-dominated solutions whenever a reference point is dominated by feasible solution(s) is to replace the selected reference point with the ideal outcome vector $z^*$. The distance components are then defined as:')
    st.latex(r'''        \Delta_k(x) = \frac{z_k^{*} - z_k(x)}{z_k^{*} - z_k^{nad}}.''')
    st.write('Adding a constraint to the Choquet program that only allows $z_i(x)$ to be lower or equal to the corresponding reference point component subsequently allows identifying all solutions that dominates the originally selected reference point:')         
    st.latex(r'''\begin{array}{rl}
         \min &  {\displaystyle \sum_{\{i\}\subseteq G} a(\{i\}) \Delta_i (x) v_i + \sum_{\{i,j\}\subseteq G} a(\{i,j\})  \beta(\{i,j\}) w_{i,j}}  \\
         subject to: & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_i (x) - My_{(\{i,j\}),1}}, \;\; \forall (\{i,j\}) \subseteq G\\
        & {\displaystyle \beta(\{i,j\}) \geqslant \Delta_j (x) - My_{(\{i,j\}),2}}, \;\; \forall (\{i,j\}) \subseteq G\\
        & {\displaystyle y_{(\{i,j\}),1} + y_{(\{i,j\}),2} = 1}, \;\; \forall (\{i,j\}) \subseteq G \\
        & \Delta_i(x) \leqslant Mv_i, \forall (\{i\}) \subseteq G \\
        & \Delta_i(x) \geqslant M(v_i-1), \forall (\{i\}) \subseteq G \\
        & \beta(\{i,j\}) \leqslant Mw_{i,j}, \forall (\{i,j\}) \subseteq G \\
        & \beta(\{i,j\}) \geqslant M(w_{i,j}-1), \forall (\{i,j\}) \subseteq G \\
        & v_i \in {0,1}, \forall (\{i\}) \subseteq G \\
        & w_{i,j} \in {0,1}, \forall (\{i,j\}) \subseteq G \\
        & z_i(x) \leqslant z^r_i, \forall (\{i\}) \subseteq G \\
        & x \in X.
    \end{array}''')
    st.write('If there is no feasible solution to that program, the originally selected reference point can be stated as a non-dominated point in the objective space. In that case, the Choquet program is solved as proposed before, using the reference point selected by the DM.')         
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

    try:
        # try to solve the first program which is the check for dominance of the reference point
        # define variables
        bigm = 100000  # big value for the linearization of the min term in the Choquet integral
        y = m.binary_var()  # binary variable for the linearization of the min term in the Choquet integral
        min_value = m.continuous_var(lb=0)  # value for the linearization of the min term in the Choquet integral

        m.add_constraint(objective[0] <= reference_point[0], ctname='check1')
        m.add_constraint(objective[1] <= reference_point[1], ctname='check2')

        # calculate the delta for both objectives from the ideal solution to the feasible space
        delta_x1 = ((f1_ideal - objective[0]) / (f1_ideal - f1_nadir))
        delta_x2 = ((f2_ideal - objective[1]) / (f2_ideal - f2_nadir))

        # constraints to determine y (y=1 if objective[0] <= objective[1], y=0 otherwise)
        m.add_constraint(delta_x2 - delta_x1 <= bigm * y, ctname='check3')
        m.add_constraint(delta_x1 - delta_x2 <= bigm * (1 - y), ctname='check4')
        # constraints to determine the min value
        m.add_constraint(min_value <= delta_x1, ctname='check5')
        m.add_constraint(min_value <= delta_x2, ctname='check6')
        m.add_constraint(min_value >= delta_x1 - (bigm * (1 - y)), ctname='check7')
        m.add_constraint(min_value >= delta_x2 - (bigm * y), ctname='check8')

        # objective function (Choquet integral program)
        choquet = a_f1 * delta_x1 + a_f2 * delta_x2 + a_f1f2 * min_value
        m.minimize(choquet)
        m.solve()

        # extract solution values
        x1_solution_choquet = x1.solution_value
        x2_solution_choquet = x2.solution_value
        z1_solution_choquet = objective[0].solution_value
        z2_solution_choquet = objective[1].solution_value

        # show solution values
        st.subheader('Solution')
        st.write('$x_1$: ' + str(x1_solution_choquet))
        st.write('$x_2$: ' + str(x2_solution_choquet))
        st.write('$z_1(x)$: ' + str(z1_solution_choquet))
        st.write('$z_2(x)$: ' + str(z2_solution_choquet))
        st.write('min value: ' + str(min_value.solution_value))
        st.write('solution: ' + str(choquet.solution_value))
        st.write('delta1: ' + str(delta_x1.solution_value))
        st.write('delta2: ' + str(delta_x2.solution_value))
        st.write('y: ' + str(y.solution_value))

        # plot the results
        st.header('The projected feasible region')
        min_value_scatter_x = min(-10, round(reference_point[0]))
        min_value_scatter_y = min(-10, round(reference_point[1]))
        fig2, ax1 = plt.subplots()
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.xticks(range(min_value_scatter_x, 26))
        plt.yticks(range(min_value_scatter_y, 26))
        plt.grid(True)
        ax1.set_xlim([min_value_scatter_x, 25])
        ax1.set_ylim([min_value_scatter_y, 25])
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.tick_params(axis='both', which='minor', labelsize=4)
        extreme_points_z1 = [6, 1, 9, 14, 12, 6]
        extreme_points_z2 = [24, 4, -3, 4, 22, 24]
        optimizing_factor = choquet.solution_value
        schnittpunkt_z1 = f1_ideal + optimizing_factor * (f1_nadir - f1_ideal)
        schnittpunkt_z2 = f2_ideal + optimizing_factor * (f2_nadir - f2_ideal)
        z1_choquet_1 = f1_ideal
        z2_choquet_1 = f2_ideal + optimizing_factor * 1 / (a_f2 / (f2_nadir - f2_ideal))
        z1_choquet_2 = f1_ideal + optimizing_factor * 1 / (a_f1 / (f1_nadir - f1_ideal))
        z2_choquet_2 = f2_ideal
        ax1.plot(extreme_points_z1, extreme_points_z2, dashes=[6, 2], linewidth=1, label='feasible area')
        ax1.plot([-10, z1_choquet_1, schnittpunkt_z1, z1_choquet_2, z1_choquet_2],
                 [z2_choquet_1, z2_choquet_1, schnittpunkt_z2, z2_choquet_2, -10], dashes=[6, 2], c='orange',
                 linewidth=1, label='Choquet metric')
        ax1.scatter(feasible_solutions_z1, feasible_solutions_z2, s=6, c='b', label='feasible solutions')
        ax1.scatter(z1_choquet_1, z2_choquet_1, s=6, c='gray')
        ax1.scatter(z1_choquet_2, z2_choquet_2, s=6, c='gray')
        ax1.scatter(schnittpunkt_z1, schnittpunkt_z2, s=6, c='gray')
        ax1.scatter(reference_point[0], reference_point[1], color='y', label='reference point ($z^{r}$)')
        ax1.scatter(z1_solution_choquet, z2_solution_choquet, color='g', label='obtained solution')
        ax1.scatter(f1_ideal, f2_ideal, c='lightgreen', label='ideal point')
        ax1.scatter(f1_nadir, f2_nadir, c='red', label='nadir point')
        ax1.plot([f1_ideal, f1_nadir], [f2_ideal, f2_nadir], color='black', linewidth=1, dashes=[6, 2])
        ax1.plot([f1_ideal, z1_solution_choquet], [f2_ideal, z2_solution_choquet])
        ax1.legend(fontsize='x-small')
        st.pyplot(fig2)
    except:
        st.write('could not solve the first problem')

        m1 = Model(name='Bi-objective convex Pareto Front')

        # decision variables
        x1 = m1.integer_var(lb=0)
        x2 = m1.integer_var(lb=0)

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
        objective[1] = 4 * x2 - x1

        # remove the two additional constraints of the first problem (the ones added to check if there are solutions that dominate the reference point)
        m.remove_constraint('check1')
        m.remove_constraint('check2')
        m.remove_constraint('check3')
        m.remove_constraint('check4')
        m.remove_constraint('check5')
        m.remove_constraint('check6')
        m.remove_constraint('check7')
        m.remove_constraint('check8')

        # define variables and parameter
        bigm = 100000  # big value for the linearization of the min term in the Choquet integral
        y_1 = m1.binary_var()  # binary variable for the linearization of the min term in the Choquet integral
        y_2 = m1.binary_var()  # binary variable for the linearization of the min term in the Choquet integral
        y = m1.binary_var()
        min_value = m1.continuous_var(lb=0)  # value for the linearization of the min term in the Choquet integral
        w_1 = m1.binary_var() # binary variable for the linearization of the mechanism to avoid penalizing negative deviations
        w_2 = m1.binary_var() # binary variable for the linearization of the mechanism to avoid penalizing negative deviations

        # calculate the delta for both objectives from the ideal solution to the feasible space
        delta_x1 = ((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir))
        delta_x2 = ((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir))

        # constraints
        # constraint to avoid negative distances to be penalized (if w=1, there is a negative distance, 0 otherwise)
        m1.add_constraint(reference_point[0]-objective[0] <= w_1 * bigm)
        m1.add_constraint(reference_point[0]-objective[0] >= (w_1-1) * bigm)
        m1.add_constraint(reference_point[1]-objective[1] <= w_2 * bigm)
        m1.add_constraint(reference_point[1]-objective[1] >= (w_2-1) * bigm)
        # constraints to linearize the interactive part of the choquet integral

        # constraints to determine y (y=1 if delta_x1 <= delta_x2, y=0 otherwise)
        m1.add_constraint(delta_x2 - delta_x1 <= bigm * y)
        m1.add_constraint(delta_x1 - delta_x2 <= bigm * (1 - y))
        # constraints to determine the min value
        m1.add_constraint(m1.if_then(w_1 == 0, min_value <= delta_x1))
        m1.add_constraint(m1.if_then(w_2 == 0, min_value <= delta_x2))
        m1.add_constraint(m1.if_then(w_1 == 0, min_value >= delta_x1 - (bigm * (1 - y))))
        m1.add_constraint(m1.if_then(w_2 == 0, min_value >= delta_x2 - (bigm * y)))

        m1.add_constraint(m1.if_then(w_1 == 1, min_value == 0))
        m1.add_constraint(m1.if_then(w_2 == 1, min_value == 0))

        # objective function
        choquet = a_f1 * (1-w_1) * delta_x1 + a_f2 * (1-w_2) * delta_x2 + (a_f1f2 * min_value)
        m1.minimize(choquet)
        m1.solve()

        '''
        if a_f1f2 >= 0:
            m.add(m.if_then(w_1 == 0, min_value >= delta_x1-(bigm*y_1)))
            m.add(m.if_then(w_2 == 0, min_value >= delta_x2-(bigm*y_2)))
            m.add_constraint(y_1 + y_2 == 1)
        else:
            m.add(m.if_then(w_1 == 0, min_value >= 0))
            m.add(m.if_then(w_2 == 0, min_value >= 0))
        '''



        '''
        # binary variable for the linearization of the interactive part of the choquet integral
        y_1 = m.binary_var()
        y_2 = m.binary_var()
        bigm = 100000
        # binary variable for the linearization of the mechanism to avoid penalizing negative deviations
        w_1 = m.binary_var()
        w_2 = m.binary_var()
        min_value = m.continuous_var(lb=0)
        delta_x1 = ((reference_point[0]-objective[0])/(reference_point[0]-f1_nadir))
        delta_x2 = ((reference_point[1]-objective[1])/(reference_point[1]-f2_nadir))
        


        # constraints to linearize the interactive part of the choquet integral
        if a_f1f2 >= 0:
            m.add(m.if_then(w_1 == 0, min_value >= delta_x1-(bigm*y_1)))
            m.add(m.if_then(w_2 == 0, min_value >= delta_x2-(bigm*y_2)))
        else:
            m.add(m.if_then(w_1 == 0, min_value <= 0))
            m.add(m.if_then(w_2 == 0, min_value <= 0))
        #m.add(m.if_then(w_1 == 0, min_value <= delta_x1))
        #m.add(m.if_then(w_2 == 0, min_value <= delta_x2))
        #m.add_constraint(min_value >= (delta_x1-(bigm*y_1))*(1-w_1))
        #m.add_constraint(min_value >= (delta_x2-(bigm*y_2))*(1-w_2))
        #m.add_constraint(min_value <= delta_x1)
        #m.add_constraint(min_value <= delta_x2)
        m.add_constraint(y_1+y_2==1)
        '''


        # extract solution values
        x1_solution_choquet = x1.solution_value
        x2_solution_choquet = x2.solution_value
        z1_solution_choquet = objective[0].solution_value
        z2_solution_choquet = objective[1].solution_value

        # show solution values
        st.subheader('Solution')
        st.write('$x_1$: ' + str(x1_solution_choquet))
        st.write('$x_2$: ' + str(x2_solution_choquet))
        st.write('$z_1(x)$: ' + str(z1_solution_choquet))
        st.write('$z_2(x)$: ' + str(z2_solution_choquet))
        st.write('min value: ' + str(min_value.solution_value))
        st.write('solution: ' + str(choquet.solution_value))
        st.write('w1: ' + str(w_1.solution_value))
        st.write('w2: ' + str(w_2.solution_value))
        st.write('delta1: ' + str(delta_x1.solution_value))
        st.write('delta2: ' + str(delta_x2.solution_value))
        st.write('y: ' + str(y.solution_value))

        # plot the results
        st.header('The projected feasible region')
        min_value_scatter_x = min(-10,round(reference_point[0]))
        min_value_scatter_y = min(-10, round(reference_point[1]))
        fig2, ax1 = plt.subplots()
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.xticks(range(min_value_scatter_x,26))
        plt.yticks(range(min_value_scatter_y, 26))
        plt.grid(True)
        ax1.set_xlim([min_value_scatter_x, 25])
        ax1.set_ylim([min_value_scatter_y, 25])
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.tick_params(axis='both', which='minor', labelsize=4)
        extreme_points_z1 = [6, 1, 9, 14, 12, 6]
        extreme_points_z2 = [24, 4,-3, 4, 22, 24]
        optimizing_factor = choquet.solution_value
        schnittpunkt_z1 = reference_point[0] + optimizing_factor * (f1_nadir-reference_point[0])
        schnittpunkt_z2 = reference_point[1] + optimizing_factor * (f2_nadir-reference_point[1])
        z1_choquet_1 = reference_point[0]
        z2_choquet_1 = reference_point[1] + optimizing_factor * 1/(a_f2/(f2_nadir-reference_point[1]))
        z1_choquet_2 = reference_point[0] + optimizing_factor * 1/(a_f1/(f1_nadir-reference_point[0]))
        z2_choquet_2 = reference_point[1]
        ax1.plot(extreme_points_z1, extreme_points_z2, dashes=[6, 2], linewidth=1, label='feasible area')
        ax1.plot([-10, z1_choquet_1, schnittpunkt_z1, z1_choquet_2, z1_choquet_2], [z2_choquet_1, z2_choquet_1, schnittpunkt_z2, z2_choquet_2, -10], dashes=[6, 2], c='orange', linewidth=1, label='Choquet metric')
        ax1.scatter(feasible_solutions_z1, feasible_solutions_z2, s=6, c='b', label='feasible solutions')
        ax1.scatter(z1_choquet_1, z2_choquet_1, s=6, c='gray')
        ax1.scatter(z1_choquet_2, z2_choquet_2, s=6, c='gray')
        ax1.scatter(schnittpunkt_z1, schnittpunkt_z2, s=6, c='gray')
        ax1.scatter(reference_point[0], reference_point[1], color='y', label='reference point ($z^{r}$)')
        ax1.scatter(z1_solution_choquet, z2_solution_choquet, color='g', label='obtained solution')
        ax1.scatter(f1_ideal, f2_ideal, c='lightgreen', label='ideal point')
        ax1.scatter(f1_nadir, f2_nadir, c='red', label='nadir point')
        ax1.plot([reference_point[0], f1_nadir], [reference_point[1], f2_nadir], color='black', linewidth=1, dashes=[6,2])
        ax1.plot([reference_point[0], z1_solution_choquet], [reference_point[1], z2_solution_choquet])
        ax1.legend(fontsize='x-small')
        st.pyplot(fig2)



