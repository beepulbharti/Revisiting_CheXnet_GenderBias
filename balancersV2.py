import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns

from matplotlib import pyplot as plt
from copy import deepcopy

import updated_tools

class BinaryBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,
                 a_hat):
            
        # Setting the variables
        self.y = y
        self.y_ = y_
        self.a = a
        self.a_hat = a_hat
        self.U = np.sum(self.a_hat != self.a)/self.a_hat.shape[0]

        ### A
        # Getting the group for A = 0 and A = 1
        self.a_groups = np.unique(a)
        a_group_ids = [np.where(a == g)[0] for g in self.a_groups]

        # P(A=0) and P(A=1)
        self.p_a = [len(cols) / len(y) for cols in a_group_ids]

        # Calcuating the groupwise classification rates for A = 0 and A = 1
        self.a_gr_list = [updated_tools.CLFRates(self.y[i], self.y_[i]) 
                         for i in a_group_ids]
        self.a_group_rates = dict(zip(self.a_groups, self.a_gr_list))
        self.base_rates = {'r_11': self.a_group_rates[1].num_pos*self.p_a[1],
                           'r_01': self.a_group_rates[0].num_pos*self.p_a[0],
                           'r_10': self.a_group_rates[1].num_neg*self.p_a[1],
                           'r_00': self.a_group_rates[0].num_neg*self.p_a[0]}

        ### A_hat
        # Getting the group for A_hat = 0 and A_hat = 1
        self.a_hat_groups = np.unique(a_hat)
        a_hat_group_ids = [np.where(a_hat == g)[0] for g in self.a_hat_groups]

        # P(A_hat=0) and P(A_hat=1)
        self.p_a_hat = [len(cols) / len(y) for cols in a_hat_group_ids]

        # Calcuating the groupwise classification rates for A = 0 and A = 1
        self.a_hat_gr_list = [updated_tools.CLFRates(self.y[i], self.y_[i]) 
                         for i in a_hat_group_ids]
        self.a_hat_group_rates = dict(zip(self.a_hat_groups, self.a_hat_gr_list))
        self.est_base_rates = {'rh_11': self.a_hat_group_rates[1].num_pos*self.p_a_hat[1],
                               'rh_01': self.a_hat_group_rates[0].num_pos*self.p_a_hat[0],
                               'rh_10': self.a_hat_group_rates[1].num_neg*self.p_a_hat[1],
                               'rh_00': self.a_hat_group_rates[0].num_neg*self.p_a_hat[0]}
        
        ### Overall
        # And then the overall rates
        self.overall_rates = updated_tools.CLFRates(self.y, self.y_)

        # Remaining relevant variables
        self.a_hat_rates = updated_tools.CLFRates(self.a, self.a_hat)
        self.U0 = self.a_hat_rates.fnr*self.p_a[1]
        self.U1 = self.a_hat_rates.fpr*self.p_a[0]

        # c constants
        self.c_01 = self.est_base_rates['rh_01'] + self.U1 - self.U0
        self.c_00 = self.est_base_rates['rh_00'] + self.U1 - self.U0

        # k constants
        self.k_11 = self.est_base_rates['rh_11'] + self.U0 - self.U1
        self.k_10 = self.est_base_rates['rh_10'] + self.U0 - self.U1

        # Calculate true bias tpr and fpr
        self.d_tpr = self.a_gr_list[1].tpr - self.a_gr_list[0].tpr
        self.d_fpr = self.a_gr_list[1].fpr - self.a_gr_list[0].fpr

        # Calculate upper and lower bounds
        rh_11 = self.est_base_rates['rh_11']
        rh_01 = self.est_base_rates['rh_01']
        rh_10 = self.est_base_rates['rh_10']
        rh_00 = self.est_base_rates['rh_00']
        U0 = self.U0
        U1 = self.U1
        self.tpr_ub = (rh_11/self.k_11)*self.a_gr_list[1].tpr - (rh_01/self.c_01)*self.a_gr_list[0].tpr \
                  + U0*(1/self.k_11 + 1/self.c_01)
        self.tpr_lb = (rh_11/self.k_11)*self.a_gr_list[1].tpr - (rh_01/self.c_01)*self.a_gr_list[0].tpr \
                  - U1*(1/self.k_11 + 1/self.c_01)
        self.fpr_ub = (rh_10/self.k_10)*self.a_gr_list[1].fpr - (rh_00/self.c_00)*self.a_gr_list[0].fpr \
                  + U0*(1/self.k_10 + 1/self.c_00)
        self.fpr_lb = (rh_10/self.k_10)*self.a_gr_list[1].fpr - (rh_00/self.c_00)*self.a_gr_list[0].fpr \
                  - U1*(1/self.k_10 + 1/self.c_00)

    def adjust(self,
               goal='odds',
               task='opt',
               round=4,
               imbalanced = True,
               return_optima=False,
               summary=False,
               binom=False):
        
        # Establish goal?
        self.goal = goal

        # Calculating relevant parameters
        rh_11 = self.est_base_rates['rh_11']
        rh_01 = self.est_base_rates['rh_01']
        rh_10 = self.est_base_rates['rh_10']
        rh_00 = self.est_base_rates['rh_00']
        c_00 = self.c_00
        c_01 = self.c_01
        k_10 = self.k_10
        k_11 = self.k_11
        U0 = self.U0
        U1 = self.U1

        # Setting loss 
        if imbalanced == True:
            l_10 = 0.5*(1/(self.overall_rates.num_neg))
            l_01 = 0.5*(1/(self.overall_rates.num_pos))
        else:
            l_10 = 1
            l_01 = 1

        # Getting the coefficients for the linear program
        coefs = [((l_10 * g.tnr * g.num_neg - l_01 * g.fnr * g.num_pos)*self.p_a_hat[i],
               (l_10 * g.fpr * g.num_neg - l_01 * g.tpr * g.num_pos)*self.p_a_hat[i])
               for i, g in enumerate(self.a_hat_gr_list)]
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.zeros((12))
        obj_coefs[:4] = np.array(coefs).flatten()
        # print(obj_coefs)
        obj_bounds = [(0, 1)]

        # # Constraint matrix and vector for generalized linear program
        g0 = self.a_hat_gr_list[0]
        g1 = self.a_hat_gr_list[1]
        A_opt = np.zeros((10,12))
        A_opt[0,0] = (rh_00/c_00)*(1 - g0.fpr)
        A_opt[0,1] = (rh_00/c_00)*g0.fpr
        A_opt[0,2] = -(rh_10/k_10)*(1-g1.fpr)
        A_opt[0,3] = -(rh_10/k_10)*g1.fpr
        A_opt[1,0] = (rh_01/c_01)*(1 - g0.tpr)
        A_opt[1,1] = (rh_01/c_01)*g0.tpr
        A_opt[1,2] = -(rh_11/k_11)*(1-g1.tpr)
        A_opt[1,3] = -(rh_11/k_11)*g1.tpr
        A_opt[2,0], A_opt[2,1], A_opt[2,4] = (1 - g0.fpr), g0.fpr, -1
        A_opt[3,0], A_opt[3,1], A_opt[3,5] = (1 - g0.fpr), g0.fpr, 1
        A_opt[4,2], A_opt[4,3], A_opt[4,6] = (1-g1.fpr), g1.fpr, -1
        A_opt[5,2], A_opt[5,3], A_opt[5,7] = (1-g1.fpr), g1.fpr, 1
        A_opt[6,0], A_opt[6,1], A_opt[6,8] = (1 - g0.tpr), g0.tpr, -1
        A_opt[7,0], A_opt[7,1], A_opt[7,9] = (1 - g0.fpr), g0.fpr, 1
        A_opt[8,2], A_opt[8,3], A_opt[8,10] = (1-g1.tpr), g1.tpr, -1
        A_opt[9,2], A_opt[9,3], A_opt[9,11] = (1-g1.tpr), g1.tpr, 1
        b_opt = np.zeros(A_opt.shape[0])
        b_opt[0], b_opt[1] = (0.5*(U0-U1))*(1/k_10 + 1/c_00), (0.5*(U0-U1))*(1/k_11 + 1/c_01)
        b_opt[2], b_opt[3] = U0/rh_00, 1 - (U0/rh_00)
        b_opt[4], b_opt[5] = U1/rh_10, 1 - (U0/rh_10)
        b_opt[6], b_opt[7] = U0/rh_01, 1 - (U0/rh_01)
        b_opt[8], b_opt[9] = U1/rh_11, 1 - (U1/rh_11)

        # Constraint matrix and vector for fairness correction
        A = np.zeros((10,12))
        A[0,0] = (1 - g0.fpr)
        A[0,1] = g0.fpr
        A[0,2] = -(1-g1.fpr)
        A[0,3] = -g1.fpr
        A[1,0] = (1 - g0.tpr)
        A[1,1] = g0.tpr
        A[1,2] = -(1-g1.tpr)
        A[1,3] = -g1.tpr
        A[2,0], A[2,1], A[2,4] = (1 - g0.fpr), g0.fpr, -1
        A[3,0], A[3,1], A[3,5] = (1 - g0.fpr), g0.fpr, 1
        A[4,2], A[4,3], A[4,6] = (1-g1.fpr), g1.fpr, -1
        A[5,2], A[5,3], A[5,7] = (1-g1.fpr), g1.fpr, 1
        A[6,0], A[6,1], A[6,8] = (1 - g0.tpr), g0.tpr, -1
        A[7,0], A[7,1], A[7,9] = (1 - g0.fpr), g0.fpr, 1
        A[8,2], A[8,3], A[8,10] = (1-g1.tpr), g1.tpr, -1
        A[9,2], A[9,3], A[9,11] = (1-g1.tpr), g1.tpr, 1
        b = np.zeros(A.shape[0])
        b[0], b[1] = 0, 0
        b[2], b[3] = U0/rh_00, 1 - (U0/rh_00)
        b[4], b[5] = U1/rh_10, 1 - (U1/rh_10)
        b[6], b[7] = U0/rh_01, 1 - (U0/rh_01)
        b[8], b[9] = U1/rh_11, 1 - (U1/rh_11)
        
        if task == 'opt':
            self.con_A = A_opt
            self.con_b = b_opt
        else: 
            self.con_A = A
            self.con_b = b

        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=self.con_A,
                                       b_eq=self.con_b,
                                       method='highs')
        self.pya = self.opt.x[:4].reshape(len(self.a_hat_groups), 2)
        
        # Setting the adjusted predictions
        self.y_adj = updated_tools.pred_from_pya(y_=self.y_, 
                                         a=self.a,
                                         pya=self.pya, 
                                         binom=binom)
        
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - updated_tools.CLFRates(self.y, self.y_adj).acc
        cmin = self.opt.fun
        
        # Calculating the theoretical balance point in ROC space

        '''
        p0, p1 = self.pya[0][0], self.pya[0][1]
        group = self.group_rates[self.groups[0]]
        fpr = (group.tnr * p0) + (group.fpr * p1)
        tpr = (group.fnr * p0) + (group.tpr * p1)
        self.roc = (np.round(fpr, round), np.round(tpr, round))
        '''
    
        if summary:
            self.summary(org=False)
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
        
        
    def predict(self, y_, a, binom=False):
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if np.any([0 < x < 1 for x in y_]):
            group_ids = [np.where(a == g)[0] for g in self.groups]
            y_ = deepcopy(y_)
            for g, cut in enumerate(self.cuts):
                y_[group_ids[g]] = updated_tools.threshold(y_[group_ids[g]], cut)
        
        # Returning the adjusted predictions
        adj = updated_tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             s1=50,
             s2=50,
             preds=False,
             optimum=True,
             lp_lines='all', 
             palette='colorblind',
             style='white',
             xlim=(0, 1),
             ylim=(0, 1)):
            
        """Generates a variety of plots for the PredictionBalancer.
        
        Parameters
        ----------
        s1, s2 : int, default 50
            The size parameters for the unadjusted (1) and adjusted (2) ROC \
            coordinates.
        
        preds : bool, default False
            Whether to observed ROC values for the adjusted predictions (as \
            opposed to the theoretical optima).
        
        optimum : bool, default True
            Whether to plot the theoretical optima for the predictions.
        
        roc_curves : bool, default True
            Whether to plot ROC curves for the unadjusted scores, when avail.
        
        lp_lines : {'upper', 'all'}, default 'all'
            Whether to plot the convex hulls solved by the linear program.
        
        shade_hull : bool, default True
            Whether to fill the convex hulls when the LP lines are shown.
        
        chance_line : bool, default True
            Whether to plot the line ((0, 0), (1, 1))
        
        palette : str, default 'colorblind'
            Color palette to pass to Seaborn.
        
        style : str, default 'dark'
            Style argument passed to sns.set_style()
        
        alpha : float, default 0.5
            Alpha parameter for scatterplots.
        
        Returns
        -------
        A plot showing shapes were specified by the arguments.
        """
        # Setting basic plot parameters
        plt.xlim(xlim)
        plt.ylim(ylim)
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = updated_tools.group_roc_coords(self.y, 
                                             self.y_, 
                                             self.a)
        sns.scatterplot(x=orig_coords.fpr,
                        y=orig_coords.tpr,
                        hue=self.groups,
                        s=s1,
                        palette='colorblind')
        plt.legend(loc='lower right')
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = updated_tools.group_roc_coords(self.y, 
                                                self.y_adj, 
                                                self.a)
            sns.scatterplot(x=adj_coords.fpr, 
                            y=adj_coords.tpr,
                            hue=self.groups,
                            palette='colorblind',
                            marker='x',
                            legend=False,
                            s=s2,
                            alpha=1)
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting the groupwise coordinates
            group_rates = self.group_rates.values()
            group_var = np.array([[g]*3 for g in self.groups]).flatten()
            
            # Getting coordinates for the upper portions of the hulls
            upper_x = np.array([[0, g.fpr, 1] for g in group_rates]).flatten()
            upper_y = np.array([[0, g.tpr, 1] for g in group_rates]).flatten()
            upper_df = pd.DataFrame((upper_x, upper_y, group_var)).T
            upper_df.columns = ['x', 'y', 'group']
            upper_df = upper_df.astype({'x': 'float',
                                        'y': 'float',
                                        'group': 'str'})
            # Plotting the line
            sns.lineplot(x='x', 
                         y='y', 
                         hue='group', 
                         data=upper_df,
                         alpha=0.75, 
                         legend=False)
            
            # Optionally adding lower lines to complete the hulls
            if lp_lines == 'all':
                lower_x = np.array([[0, 1 - g.fpr, 1] 
                                    for g in group_rates]).flatten()
                lower_y = np.array([[0, 1 - g.tpr, 1] 
                                    for g in group_rates]).flatten()
                lower_df = pd.DataFrame((lower_x, lower_y, group_var)).T
                lower_df.columns = ['x', 'y', 'group']
                lower_df = lower_df.astype({'x': 'float',
                                            'y': 'float',
                                            'group': 'str'})
                # Plotting the line
                sns.lineplot(x='x', 
                             y='y', 
                             hue='group', 
                             data=lower_df,
                             alpha=0.75, 
                             legend=False)       
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.roc is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            elif 'odds' in self.goal:
                plt.scatter(self.roc[0],
                                self.roc[1],
                                marker='x',
                                color='black')
        
        plt.show()
    
    def summary(self, org=True, adj=True):
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        if org:
            org_coords = updated_tools.group_roc_coords(self.y, self.y_, self.a)
            org_loss = 1 - self.overall_rates.acc
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            adj_coords = updated_tools.group_roc_coords(self.y, self.y_adj, self.a)
            adj_loss = 1 - updated_tools.CLFRates(self.y, self.y_adj).acc
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %adj_loss)
