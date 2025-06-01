from contour import pencil, pencil2, fan, fan2, fan3, combine
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from helper_functions import weighted_norm, get_intensity

#Below imports are used for MCMC
import timeit
import emcee, corner
from scipy.stats import beta


def MCMC_not_combine(data1, data2, mirror, func, n_walkers, cycles, discard):
    """Standard MCMC for a non combine function since combine has more parameters.

    Args:
        data1 (list): normalized data for pole 1
        data2 (list): normalized data for pole 2
        mirror (int): mirror criteria, 1 for no mirror, -1 for mirror
        func (function): specified function, this does not allow for combined
        n_walkers (int): number of walkers
        cycles (int): number of cycles
        discard (int): number of discards, shall be less than the number of cycles
    """    
    def log_likelihood_gauss(params, data1, yerr1, data2, yerr2):
        """log likelihood for a gaussian distrubiution.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: log likelihood for the parameters with a gaussian distrubiution.
        """        
        i, pa, offset, phs, i_offset, az_offset, r, gamma = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma, func = func)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), mirror*i_offset, mirror*az_offset, r, gamma, func=func)
        sigma2_1 = yerr1 ** 2
        sigma2_2 = yerr2 ** 2
        
        log_like1 = -0.5 * np.sum((data1 - model_y) ** 2 / sigma2_1 + np.log(sigma2_1))
        log_like2 = -0.5 * np.sum((data2 - model_y2) ** 2 / sigma2_2 + np.log(sigma2_2))
        
        return log_like1 + log_like2

    def log_likelihood_beta(params, data1, yerr1, data2, yerr2):
        """log likelihood for a beta distrubiution. UNUSED

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: log likelihood for the parameters with a beta distrubiution.
        """        
        i, pa, offset, phs, i_offset, az_offset, r, gamma = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma, func=func)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), i_offset, az_offset, r, gamma, func = func)
        
        kappa_1 = 1 /(yerr1 ** 2)
        kappa_2 = 1 /(yerr2 ** 2)
        
        alpha1 = model_y * kappa_1
        alpha2 = model_y2 * kappa_2
        
        beta1 = (1-model_y) * kappa_1
        beta2 = (1-model_y2) * kappa_2
        
        alpha1 = np.clip(alpha1, 1e-3, None)
        beta1 = np.clip(beta1, 1e-3, None)
        alpha2 = np.clip(alpha2, 1e-3, None)
        beta2 = np.clip(beta2, 1e-3, None)
        
        log_like1 = np.sum(beta.logpdf(data1, alpha1, beta1))
        log_like2 = np.sum(beta.logpdf(data2, alpha2, beta2))
        
        return log_like1 + log_like2

    # Define prior distribution-
    def log_prior(params):
        """returns the log of the prior, rejects if parameters are out of bonds

        Args:
            params (list): list of parameters

        Returns:
            int: 0 for allowed -inf for out of bounds (rejection) 
        """        
        i, pa, offset, phs, i_offset, az_offset, r, gamma = params
        if (-np.pi/2 < i < np.pi/2 and -np.pi/2 < pa < np.pi/2 and 0 < offset < np.pi and
            0 < phs < 2*np.pi and -np.pi/4 < i_offset < np.pi/4 and -np.pi/4 < az_offset < np.pi/4 and
            3 < r < 3.05 and 1 < gamma < 5):
            return 0.0  # Flat prior
        return -np.inf  # Reject if out of bounds

    # Define total log probability
    def log_probability(params, data1, yerr1, data2, yerr2):
        """log probability.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: likelihood for the parameters.
        """    
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood_gauss(params, data1, yerr1, data2, yerr2)

    # MCMC function
    def run_mcmc(data1, yerr_1, data2, yerr_2):
        """Runs the mcmc simulation.

        Args:
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            list: list of best fit parameters
        """            
        start_time = timeit.default_timer()
    
        # Initial guess taken from 1 row in softmax.csv for f with mirror
        initial_params = [np.deg2rad(-59.784), np.deg2rad(89.308), np.deg2rad(130.108), np.deg2rad(158.973), np.deg2rad(-30.074), np.deg2rad(-25.955), 3.037, 4.209]
        
        # Initial guess taken from 77* row in softmax.csv for f with no mirror
        #initial_params =  [0.71332136,  1.55391952,  2.93255578,  3.30362209,  0.57994181, -0.18205942, 3.03621336, 1.00052988]
        
        # Initial guess taken from 78* row in softmax.csv for h1 with mirror
        #initial_params =  [ 0.63257113, -0.58716608,  0.47653356,  3.27294592, -0.21317315, -0.53855258, 3.03665307,  4.946509]
        
        # Initial guess taken from 79* row in softmax.csv for h1 with no mirror
        #initial_params =  [ 0.26828954, -1.23358778,  2.51852683,  0.15467617,  0.61940721,  0.69210199, 3.03190642,  4.99679765]
        
        nwalkers = n_walkers
        ndim = len(initial_params)
        
        # Small random perturbation around initial guess
        initial_pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
        
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(data1, yerr_1, data2, yerr_2))
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, cycles, progress=True)
        
        # Get samples after burn-in
        samples = sampler.get_chain(discard=discard, flat=True)
        best_fit_params = np.mean(samples, axis=0)
        
        samples_deg = np.copy(samples)
        samples_deg[:, :6] = np.rad2deg(samples_deg[:, :6])
        best_fit_params_deg = np.copy(best_fit_params)
        best_fit_params_deg[:6] = np.rad2deg(best_fit_params_deg[:6])
        
        print("Best Fit Parameters (MCMC):", best_fit_params_deg)
        
        # Compute best-fit intensity
        intensities_best = get_intensity(*best_fit_params, func = func)
        intensities_best2 = get_intensity(best_fit_params[0], best_fit_params[1], np.pi - best_fit_params[2], (np.pi + best_fit_params[3]) % (2*np.pi) , 
                                         mirror*best_fit_params[4], mirror*best_fit_params[5], best_fit_params[6], best_fit_params[7], func = func)
        
        # Plot results
        xarray = np.linspace(0, 1, 32)
        plt.figure()
        plt.plot(xarray, data1, label="Observed Data", color="blue", linestyle="dashed")
        plt.plot(xarray, intensities_best, label="Best Fit", color="blue")
        plt.plot(xarray, data2, label="Observed Data", color="red", linestyle="dashed")
        plt.plot(xarray, intensities_best2, label="Best Fit", color="red")
        
        plt.ylim(0,1)
        
        plt.xlabel("X")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot corner plot
        
        rcParams["font.size"] = 8
        
        labels=["inclination", "PA", "magnetic offset", "phase shift", "θ_off", "φ_offset", "radius", "γ"]
        
        fig = plt.figure(figsize=(12, 12))  # adjust size as needed
        corner.corner(samples_deg, labels=labels, 
                    truths=best_fit_params_deg, 
                    show_titles=True, 
                    title_kwargs={"fontsize": 10}, 
                    label_kwargs={"fontsize": 8},
                    fig=fig)

        # Now hide the x-axis labels for the bottom row
        ndim = len(labels)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for i in range(ndim):
            ax = axes[-1, i]  # bottom row
            ax.set_xlabel("")  # remove label
        
        plt.tight_layout(pad = 10)
        plt.show()
        
        
        return best_fit_params

    yerr_1 = np.clip(np.full_like(data1, 0.1), 1e-4, None)  # Prevent division by zero
    yerr_2 = np.clip(np.full_like(data2, 0.1), 1e-4, None)  # Prevent division by zero
    
    run_mcmc(data1, yerr_1, data2, yerr_2)


def MCMC_not_combine_constrained(data1, data2, mirror, func, n_walkers, cycles, discard):
    """Constrained MCMC for a non combine function with incorperated constrains from Tsygankov et al. (2022).

    Args:
        data1 (list): normalized data for pole 1
        data2 (list): normalized data for pole 2
        mirror (int): mirror criteria, 1 for no mirror, -1 for mirror
        func (function): specified function, this does not allow for combined
        n_walkers (int): number of walkers
        cycles (int): number of cycles
        discard (int): number of discards, shall be less than the number of cycles
    """    
    def log_likelihood_gauss(params, data1, yerr1, data2, yerr2):
        """log likelihood for a gaussian distrubiution.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: log likelihood for the parameters with a gaussian distrubiution.
        """  
        phs, i_offset, az_offset, r, gamma = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma, func = func)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), mirror*i_offset, mirror*az_offset, r, gamma, func=func)
        sigma2_1 = yerr1 ** 2
        sigma2_2 = yerr2 ** 2
        
        log_like1 = -0.5 * np.sum((data1 - model_y) ** 2 / sigma2_1 + np.log(sigma2_1))
        log_like2 = -0.5 * np.sum((data2 - model_y2) ** 2 / sigma2_2 + np.log(sigma2_2))
        
        return log_like1 + log_like2


    # Define prior distribution-
    def log_prior(params):
        """returns the log of the prior, rejects if parameters are out of bonds

        Args:
            params (list): list of parameters

        Returns:
            int: 0 for allowed -inf for out of bounds (rejection) 
        """  
        phs, i_offset, az_offset, r, gamma = params
        if (0 < phs < 2*np.pi and -np.pi/4 < i_offset < np.pi/4 and -np.pi/4 < az_offset < np.pi/4 and
            3 < r < 3.05 and 1 < gamma < 5):
            return 0.0  # Flat prior
        return -np.inf  # Reject if out of bounds

    # Define total log probability
    def log_probability(params, data1, yerr1, data2, yerr2):
        """log probability.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: likelihood for the parameters.
        """    
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood_gauss(params, data1, yerr1, data2, yerr2)

    # MCMC function
    def run_mcmc(data1, yerr_1, data2, yerr_2):
        """Runs the mcmc simulation.

        Args:
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            list: list of best fit parameters
        """          
        global i, pa, offset 
        i, pa, offset = np.deg2rad(109.8), np.deg2rad(49), np.deg2rad(17)
        start_time = timeit.default_timer()
        
        initial_params = [np.deg2rad(158.973), np.deg2rad(-30.074), np.deg2rad(-25.955), 3.037, 4.209]
        
        nwalkers = n_walkers
        ndim = len(initial_params)
        
        # Small random perturbation around initial guess
        initial_pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
        
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(data1, yerr_1, data2, yerr_2))
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, cycles, progress=True)
        
        # Get samples after burn-in
        samples = sampler.get_chain(discard=discard, flat=True)
        best_fit_params = np.mean(samples, axis=0)
        
        samples_deg = np.copy(samples)
        samples_deg[:, :3] = np.rad2deg(samples_deg[:, :3])
        best_fit_params_deg = np.copy(best_fit_params)
        best_fit_params_deg[:3] = np.rad2deg(best_fit_params_deg[:3])
        
        print("Best Fit Parameters (MCMC):", best_fit_params_deg)
        
        # Compute best-fit intensity
        intensities_best = get_intensity(i, pa, offset, best_fit_params[0], 
                                        best_fit_params[1], best_fit_params[2], best_fit_params[3], best_fit_params[4], func = func)
        intensities_best2 = get_intensity(i, pa, np.pi - offset, (np.pi + best_fit_params[0]) % (2*np.pi) , 
                                         mirror*best_fit_params[1], mirror*best_fit_params[2], best_fit_params[3], best_fit_params[4], func = func)
        
        # Plot results
        xarray = np.linspace(0, 1, 32)
        plt.figure()
        plt.plot(xarray, data1, label="Observed Data", color="blue", linestyle="dashed")
        plt.plot(xarray, intensities_best, label="Best Fit", color="blue")
        plt.plot(xarray, data2, label="Observed Data", color="red", linestyle="dashed")
        plt.plot(xarray, intensities_best2, label="Best Fit", color="red")
        
        plt.ylim(0,1)
        
        plt.xlabel("Phase")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot corner plot
        
        rcParams["font.size"] = 8
        
        # Exclude first 3 parameters (inclination, PA, magnetic offset)
        samples_deg_trimmed = samples_deg
        best_fit_params_deg_trimmed = best_fit_params_deg
        labels_trimmed = ["phase shift", "θ_off", "φ_offset", "radius", "γ"]

        fig = plt.figure(figsize=(30, 30))  # adjust size as needed
        corner.corner(samples_deg_trimmed, labels=labels_trimmed, 
                    truths=best_fit_params_deg_trimmed, 
                    show_titles=True, 
                    title_kwargs={"fontsize": 12}, 
                    label_kwargs={"fontsize": 8},
                    fig=fig)

        # Now hide the x-axis labels for the bottom row
        ndim = len(labels_trimmed)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for i in range(ndim):
            ax = axes[-1, i]  # bottom row
            ax.set_xlabel("")  # remove label
        
        plt.tight_layout(pad = 10)
        plt.show()
        
        
        return best_fit_params

    yerr_1 = np.clip(np.full_like(data1, 0.1), 1e-4, None)  # Prevent division by zero
    yerr_2 = np.clip(np.full_like(data2, 0.1), 1e-4, None)  # Prevent division by zero
    
    run_mcmc(data1, yerr_1, data2, yerr_2)
    



def MCMC_combine(data1, data2, mirror, func, n_walkers, cycles, discard):
    """Standard MCMC for a combine function is a different function from MCMC_not_combine 
    because combine has more parameters

    Args:
        data1 (list): normalized data for pole 1
        data2 (list): normalized data for pole 2
        mirror (int): mirror criteria, 1 for no mirror, -1 for mirror
        func (function): specified function, this shall be combined
        n_walkers (int): number of walkers
        cycles (int): number of cycles
        discard (int): number of discards, shall be less than the number of cycles
    """   
    def log_likelihood_gauss(params, data1, yerr1, data2, yerr2):
        """log likelihood for a gaussian distrubiution.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: log likelihood for the parameters with a gaussian distrubiution.
        """  
        i, pa, offset, phs, i_offset, az_offset, r, gamma, gamma2, w1 = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma, gamma2=gamma2, weight1=w1, weight2= 1-w1, func = func)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), mirror*i_offset, mirror*az_offset, r, gamma, gamma2=gamma2, weight1=w1, weight2= 1-w1, func=func)
        sigma2_1 = yerr1 ** 2
        sigma2_2 = yerr2 ** 2
        
        log_like1 = -0.5 * np.sum((data1 - model_y) ** 2 / sigma2_1 + np.log(sigma2_1))
        log_like2 = -0.5 * np.sum((data2 - model_y2) ** 2 / sigma2_2 + np.log(sigma2_2))
        
        return log_like1 + log_like2

    def log_likelihood_beta(params, data1, yerr1, data2, yerr2):
        i, pa, offset, phs, i_offset, az_offset, r, gamma, gamma2, w1 = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), i_offset, az_offset, r, gamma, gamma2=gamma2, weight1=w1)
        
        kappa_1 = 1 /(yerr1 ** 2)
        kappa_2 = 1 /(yerr2 ** 2)
        
        alpha1 = model_y * kappa_1
        alpha2 = model_y2 * kappa_2
        
        beta1 = (1-model_y) * kappa_1
        beta2 = (1-model_y2) * kappa_2
        
        alpha1 = np.clip(alpha1, 1e-3, None)
        beta1 = np.clip(beta1, 1e-3, None)
        alpha2 = np.clip(alpha2, 1e-3, None)
        beta2 = np.clip(beta2, 1e-3, None)
        
        log_like1 = np.sum(beta.logpdf(data1, alpha1, beta1))
        log_like2 = np.sum(beta.logpdf(data2, alpha2, beta2))
        
        return log_like1 + log_like2

    # Define prior distribution-
    def log_prior(params):
        """returns the log of the prior, rejects if parameters are out of bonds

        Args:
            params (list): list of parameters

        Returns:
            int: 0 for allowed -inf for out of bounds (rejection) 
        """  
        i, pa, offset, phs, i_offset, az_offset, r, gamma, gamma2, w1 = params
        if (-np.pi/2 < i < np.pi/2 and -np.pi/2 < pa < np.pi/2 and 0 < offset < np.pi and
            0 < phs < 2*np.pi and -np.pi/4 < i_offset < np.pi/4 and -np.pi/4 < az_offset < np.pi/4 and
            3 < r < 3.05 and 1 < gamma < 5 and 1 < gamma2 < 5 and 0 < w1 < 1):
            return 0.0  # Flat prior
        return -np.inf  # Reject if out of bounds

    # Define total log probability
    def log_probability(params, data1, yerr1, data2, yerr2):
        """log probability.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: likelihood for the parameters.
        """    
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood_gauss(params, data1, yerr1, data2, yerr2)

    # MCMC function
    def run_mcmc(data1, yerr_1, data2, yerr_2):
        """Runs the mcmc simulation.

        Args:
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            list: list of best fit parameters
        """          
        start_time = timeit.default_timer()
    
        # Initial guess taken from 74* row in softmax.csv this is for combine with mirror
        #initial_params = [0.59758459, 0.75492051, 1.40387786, 3.0476457,
        #                -0.62633691, 0.50094771, 3.00943873,4.68715862, 2.08917393, 0.86554303]

        # Initial guess taken from 75* row in softmax.csv this is for combine with mirror
        # This seed produces unwanted spike
        #initial_params = [ 1.43065562,  0.68636012,  0.41352492,  0.68686504, -0.25090363,  0.50583791,
        #3.04439134,  4.99843072,  4.99997863, 0.60024302]
        
        # Initial guess taken from 80* row in softmax.csv this is for combine with mirror
        #initial_params = [-6.58587469e-01 , -3.01140001e-03,  6.58594869e-01,  2.41148802e-06,
        #3.05062272e-02,  7.85397877e-01,  3.00000003e+00,  4.99956009e+00,
        #4.99998660e+00, 1.75492157e-01 ]        
        
        # Initial guess taken from 76* row in softmax.csv this is for combine with no mirror
        initial_params = [ 1.38365449,  0.86976742 , 1.15881604,  4.6597003, -0.50881776,  -0.29144277,
        3.03917149,  4.9825829,   3.90944986, 0.73102077]
                

        nwalkers = n_walkers
        ndim = len(initial_params)
        
        # Small random perturbation around initial guess
        initial_pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
        
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(data1, yerr_1, data2, yerr_2))
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, cycles, progress=True)
        
        # Get samples after burn-in
        samples = sampler.get_chain(discard=discard, flat=True)
        best_fit_params = np.mean(samples, axis=0)
        
        samples_deg = np.copy(samples)
        samples_deg[:, :6] = np.rad2deg(samples_deg[:, :6])
        best_fit_params_deg = np.copy(best_fit_params)
        best_fit_params_deg[:6] = np.rad2deg(best_fit_params_deg[:6])
        
        print("Best Fit Parameters (MCMC):", best_fit_params_deg)
        
        # Compute best-fit intensity
        intensities_best = get_intensity(best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], 
                                         best_fit_params[4], best_fit_params[5], best_fit_params[6], best_fit_params[7], gamma2=best_fit_params[8], 
                                         weight1=best_fit_params[9], weight2= 1-best_fit_params[9], func = func)
        
        intensities_best2 = get_intensity(best_fit_params[0], best_fit_params[1], np.pi - best_fit_params[2], ((np.pi + best_fit_params[3]) % (2*np.pi)) , 
                                         mirror*best_fit_params[4], mirror*best_fit_params[5], best_fit_params[6], best_fit_params[7], gamma2=best_fit_params[8], 
                                         weight1=best_fit_params[9], weight2= 1-best_fit_params[9], func = func)
        
        # Plot results
        xarray = np.linspace(0, 1, 32)
        plt.figure()
        plt.plot(xarray, data1, label="Observed Data", color="blue", linestyle="dashed")
        plt.plot(xarray, intensities_best, label="Best Fit", color="blue")
        plt.plot(xarray, data2, label="Observed Data", color="red", linestyle="dashed")
        plt.plot(xarray, intensities_best2, label="Best Fit", color="red")
        
        plt.ylim(0,1)
        
        plt.xlabel("X")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot corner plot
        
        rcParams["font.size"] = 8
        
        labels=["inclination", "PA", "magnetic offset", "phase shift", "θ_off", "φ_offset", "radius", "γ", "γ_2", "weight"]
        
        fig = plt.figure(figsize=(12, 12))  # adjust size as needed
        corner.corner(samples_deg, labels=labels, 
                    truths=best_fit_params_deg, 
                    show_titles=True, 
                    title_kwargs={"fontsize": 10}, 
                    label_kwargs={"fontsize": 8},
                    fig=fig)

        # Now hide the x-axis labels for the bottom row
        ndim = len(labels)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for i in range(ndim):
            ax = axes[-1, i]  # bottom row
            ax.set_xlabel("")  # remove label
        
        plt.tight_layout(pad = 10)
        plt.show()
        
        
        return best_fit_params

    yerr_1 = np.clip(np.full_like(data1, 0.1), 1e-4, None)  # Prevent division by zero
    yerr_2 = np.clip(np.full_like(data2, 0.1), 1e-4, None)  # Prevent division by zero
    
    run_mcmc(data1, yerr_1, data2, yerr_2)


def MCMC_combine_constrained(data1, data2, mirror, func, n_walkers, cycles, discard):
    """Constrained MCMC for a combine function is a different function from MCMC_not_combine 
    because combine has more parameters incorperatred constrains from Tsygankov et al. (2022).

    Args:
        data1 (list): normalized data for pole 1
        data2 (list): normalized data for pole 2
        mirror (int): mirror criteria, 1 for no mirror, -1 for mirror
        func (function): specified function, this shall be combined
        n_walkers (int): number of walkers
        cycles (int): number of cycles
        discard (int): number of discards, shall be less than the number of cycles
    """   
    def log_likelihood_gauss(params, data1, yerr1, data2, yerr2):
        """log likelihood for a gaussian distrubiution.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: log likelihood for the parameters with a gaussian distrubiution.
        """  
        phs, i_offset, az_offset, r, gamma, gamma2, w1 = params
        model_y = get_intensity(i, pa, offset, phs, i_offset, az_offset, r, gamma, gamma2=gamma2, weight1=w1, weight2= 1-w1, func = func)
        model_y2 = get_intensity(i, pa, (np.pi - offset), (np.pi + phs) % (2*np.pi), mirror*i_offset, mirror*az_offset, r, gamma, gamma2=gamma2, weight1=w1, weight2= 1-w1, func=func)
        sigma2_1 = yerr1 ** 2
        sigma2_2 = yerr2 ** 2
        
        log_like1 = -0.5 * np.sum((data1 - model_y) ** 2 / sigma2_1 + np.log(sigma2_1))
        log_like2 = -0.5 * np.sum((data2 - model_y2) ** 2 / sigma2_2 + np.log(sigma2_2))
        
        return log_like1 + log_like2

    # Define prior distribution-
    def log_prior(params):
        """returns the log of the prior, rejects if parameters are out of bonds

        Args:
            params (list): list of parameters

        Returns:
            int: 0 for allowed -inf for out of bounds (rejection) 
        """  
        phs, i_offset, az_offset, r, gamma, gamma2, w1 = params
        if (
            0 < phs < 2*np.pi and -np.pi/4 < i_offset < np.pi/4 and -np.pi/4 < az_offset < np.pi/4 and
            3 < r < 3.05 and 1 < gamma < 5 and 1 < gamma2 < 5 and 0 < w1 < 1):
            return 0.0  # Flat prior
        return -np.inf  # Reject if out of bounds

    # Define total log probability
    def log_probability(params, data1, yerr1, data2, yerr2):
        """log probability.

        Args:
            params (list): parameters
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            float: likelihood for the parameters.
        """    
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood_gauss(params, data1, yerr1, data2, yerr2)

    # MCMC function
    def run_mcmc(data1, yerr_1, data2, yerr_2):
        """Runs the mcmc simulation.

        Args:
            data1 (list): normalized data for pole 1
            yerr1 (list): the allowed error for pole 1
            data2 (list): normalized data for pole 2
            yerr2 (list): the allowed error for pole 2

        Returns:
            list: list of best fit parameters
        """          
        start_time = timeit.default_timer()
        global i, pa, offset 
        i, pa, offset = np.deg2rad(109.8), np.deg2rad(49), np.deg2rad(17)
        # Initial guess taken from 74* row in softmax.csv this is for combine with mirror
        #initial_params = [0.59758459, 0.75492051, 1.40387786, 3.0476457,
        #                -0.62633691, 0.50094771, 3.00943873,4.68715862, 2.08917393, 0.86554303]

        # Initial guess taken from 75* row in softmax.csv this is for combine with mirror
        # This seed produces unwanted spike
        #initial_params = [ 1.43065562,  0.68636012,  0.41352492,  0.68686504, -0.25090363,  0.50583791,
        #3.04439134,  4.99843072,  4.99997863, 0.60024302]
        
        # Initial guess taken from 80* row in softmax.csv this is for combine with mirror
        #initial_params = [-6.58587469e-01 , -3.01140001e-03,  6.58594869e-01,  2.41148802e-06,
        #3.05062272e-02,  7.85397877e-01,  3.00000003e+00,  4.99956009e+00,
        #4.99998660e+00, 1.75492157e-01 ]        
        
        # Initial guess taken from 76* row in softmax.csv this is for combine with no mirror
        initial_params = [4.6597003, -0.50881776,  -0.29144277,
        3.03917149,  4.9825829,   3.90944986, 0.73102077]
                

        nwalkers = n_walkers
        ndim = len(initial_params)
        
        # Small random perturbation around initial guess
        initial_pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)
        
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(data1, yerr_1, data2, yerr_2))
        print("Running MCMC...")
        sampler.run_mcmc(initial_pos, cycles, progress=True)
        
        # Get samples after burn-in
        samples = sampler.get_chain(discard=discard, flat=True)
        best_fit_params = np.mean(samples, axis=0)
        
        samples_deg = np.copy(samples)
        samples_deg[:, :3] = np.rad2deg(samples_deg[:, :3])
        best_fit_params_deg = np.copy(best_fit_params)
        best_fit_params_deg[:3] = np.rad2deg(best_fit_params_deg[:3])
        
        print("Best Fit Parameters (MCMC):", best_fit_params_deg)
        
        # Compute best-fit intensity
        intensities_best = get_intensity(i, pa, offset, best_fit_params[0], 
                                         best_fit_params[1], best_fit_params[2], best_fit_params[3], best_fit_params[4], gamma2=best_fit_params[5], 
                                         weight1=best_fit_params[6], weight2= 1-best_fit_params[6], func = func)
        
        intensities_best2 = get_intensity(i, pa, offset, ((np.pi + best_fit_params[0]) % (2*np.pi)) , 
                                         mirror*best_fit_params[1], mirror*best_fit_params[2], best_fit_params[3], best_fit_params[4], gamma2=best_fit_params[5], 
                                         weight1=best_fit_params[6], weight2= 1-best_fit_params[6], func = func)
        
        # Plot results
        xarray = np.linspace(0, 1, 32)
        plt.figure()
        plt.plot(xarray, data1, label="Observed Data", color="blue", linestyle="dashed")
        plt.plot(xarray, intensities_best, label="Best Fit", color="blue")
        plt.plot(xarray, data2, label="Observed Data", color="red", linestyle="dashed")
        plt.plot(xarray, intensities_best2, label="Best Fit", color="red")
        
        plt.ylim(0,1)
        
        plt.xlabel("X")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot corner plot
        
        rcParams["font.size"] = 8
        
        labels=["phase shift", "θ_off", "φ_offset", "radius", "γ", "γ_2", "weight"]
        
        fig = plt.figure(figsize=(36, 36))  # adjust size as needed
        corner.corner(samples_deg, labels=labels, 
                    truths=best_fit_params_deg, 
                    show_titles=True, 
                    title_kwargs={"fontsize": 10}, 
                    label_kwargs={"fontsize": 8},
                    fig=fig)

        # Now hide the x-axis labels for the bottom row
        ndim = len(labels)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for i in range(ndim):
            ax = axes[-1, i]  # bottom row
            ax.set_xlabel("")  # remove label
        
        plt.tight_layout(pad = 10)
        plt.show()
        
        
        return best_fit_params

    yerr_1 = np.clip(np.full_like(data1, 0.1), 1e-4, None)  # Prevent division by zero
    yerr_2 = np.clip(np.full_like(data2, 0.1), 1e-4, None)  # Prevent division by zero
    
    run_mcmc(data1, yerr_1, data2, yerr_2)

def main():
    """Main run file of the MCMC simulation, has the data sets, normalizes them and 
    then calls the specified MCMC simulation with a number of walkers, cycles, discards, function and mirror criteria
    """    
    d1 = np.array([2.08943366e+02, 2.19458930e+02, 2.24270196e+02, 2.21582811e+02,
                   2.11904408e+02, 1.89629016e+02, 1.69920199e+02, 1.25652313e+02,
                   8.50368827e+01, 7.87287160e+01, 6.99192303e+01, 5.68317891e+01,
                   3.12415676e+01, 7.26903682e+01, 6.02122557e+01, 3.28354915e-05,
                   2.07074634e+01, 2.03920429e+02, 4.84004943e+02, 7.17241518e+02,
                   8.87482781e+02, 9.76466087e+02, 9.54130525e+02, 8.21581334e+02,
                   6.19085004e+02, 4.59888725e+02, 3.71295449e+02, 3.15689818e+02,
                   2.66624161e+02, 2.25634674e+02, 2.10376988e+02, 2.07897609e+02])
    d2 = np.array([170.6203983, 162.72957678, 150.83541175, 138.78669326, 122.24861362,
                   113.49124518, 91.34073072, 83.93452248, 106.68171974, 115.37107808,
                   123.3759738, 148.31675151, 249.29131364, 339.91796409, 463.67402264,
                   680.03072894, 891.37404325, 909.39742209, 720.29686162, 501.44087283,
                   298.15601013, 126.65120765, 27.76344419, 3.59881618, 51.63147899,
                   102.73432173, 122.70552028, 125.26718809, 131.90538957, 148.83504852,
                   157.45255021, 165.55309996])
    d1normalized, d2normalized = weighted_norm(d1, d2)
    
    mirror = -1 # -1 for mirror, 1 for no mirror
    function = combine
    
    n_walkers = 1024
    cycles = 5000
    discard = 1000
    
    MCMC_combine_constrained(d1normalized, d2normalized, mirror, function, n_walkers, cycles, discard)
    #MCMC_combine(d1normalized, d2normalized, mirror, n_walkers, cycles, discard)


if __name__ == '__main__':
    main()


