# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:17:04 2020

@author: chris
"""

import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os

from tenpy.networks.mps import MPS, build_initial_state#,from_product_state
import tenpy.networks.mpo as MPO
import tenpy.algorithms.dmrg as dmrg
import tenpy.models.hofstadter_v5 as mod
import tenpy.tools.misc as misc

def get_filling(run_par, L):
    # Compute filling fractions based on set quantum numbers.
    # Nup_MPS = int((run_par['Ntot_MPS'] + run_par['2Sz_MPS'])/2)
    # Ndn_MPS = run_par['Ntot_MPS'] - Nup_MPS
    # Ntot_MPS = Nup_MPS + Ndn_MPS
    # if Ntot_MPS != run_par['Ntot_MPS'] or Nup_MPS-Ndn_MPS != run_par['2Sz_MPS'] or Nup_MPS < 0 or Ndn_MPS < 0:
    #     print('Nup_MPS=', Nup_MPS, ',  Ndn_MPS=', Ndn_MPS, ',  Ntot_MPS=',
    #           run_par['Ntot_MPS'], ',  2Sz_MPS=', run_par['2Sz_MPS'])
    #     raise ValueError("inconsistency in the particle number setting")
    Nup_MPS = run_par['Nup_MPS']
    Ndn_MPS = run_par['Ndn_MPS']
    upfilling, dnfilling = float(Nup_MPS)/L, float(Ndn_MPS)/L
    return upfilling, dnfilling

def random_list(length,upfilling,dnfilling):
    """this function generates random occupations for an initial state
    of the requested length 
       with densities of upfilling, dnfilling for the up and down spins,
       respectively
       A corresponding product state MPS is returned."""
    uplist = build_initial_state(size=length,states=(0,1),filling=(1-upfilling,upfilling))
    dnlist = build_initial_state(size=length,states=(0,1),filling=(1-dnfilling,dnfilling))
    prod=[]
    for i in range(length):
        if (uplist[i]==0 and dnlist[i]==0):
            prod.append("empty")
        elif (uplist[i]==1 and dnlist[i]==0):
            prod.append("up")
        elif (uplist[i]==0 and dnlist[i]==1):
            prod.append("down")
        elif (uplist[i]==1 and dnlist[i]==1):
            prod.append("full")
    return prod

def cleanup_output(out):
	"""Remove the LP and RP environments from out, if they exist.

	Args:
	    out (dict): original dictionary

	Returns:
	    dict: out, but with LP and RP deleted.
	"""
	try:
		del out['LP']
	except (KeyError, TypeError) as e:
		pass
	try:
		del out['RP']
	except (KeyError, TypeError) as e:
		pass
	return out

def dmrg_wrapper(model_par, sim_par, run_par, args, psi=None, sim_filename=None, reload_sim=False):
    """Run DMRG ("Walk this way...").

	Args:
	    model_par (dict): Model parameters
	    args (Namespace): All command line options
	    psi (MPS, optional): Initial state. If None, random state is iinitialized based on model_par and args.

	Returns:
	    psi (MPS): Ground state
	    out (dict): DMRG output
	    num_ini (array): Density profile for the initial state
	    M (model): model

	"""
    print ("chi's in use:", sim_par['CHI_LIST'])
    print ("minimum no. of sweeps:", sim_par['MIN_STEPS'])
    print ('dmrg_wrapper has bc={}'.format(model_par['bc']))

	# Check if there is already a data file from a finished run
    try:
        with open('fh_dmrg_' + sim_par['identifier'] + '.dat', 'rb') as f:
            a = pickle.load(f)
        if data['out']['shelve_flag']:
            print ("Simulation was shelved. Restarting soon.")
        else:
            print ("Finished data file found. Skipping.")
            return None, {'skip':True}, None, None, None  # Not very clean but haven't found anything better
    except IOError:
        print ("No data file found. Starting new sim.")
        pass

    if reload_sim:
        with open(sim_filename + '.sim', 'rb') as f:
            save_sim = pickle.load(f)
        sim_par['STARTING_ENV_FROM_PSI'] = 0  # Make sure to use old environments.
        sim_par['CHI_LIST'] = {0:max(sim_par['CHI_LIST'].values())}  # Assume saved sim had reached max chi.
        sim_par['MIN_STEPS'] = 20
        sim_par['LP'] = save_sim['LP']
        sim_par['RP'] = save_sim['RP']
        psi = save_sim['psi']
        M = save_sim['M']
        num_ini = save_sim['num_ini']
        initial_state = save_sim['initial_state']
    else:
        M = mod.HofstadterFermions(model_par)
        initial_state = build_initial_state(M.lat.N_sites, (0,1), (1-run_par['filling'],run_par['filling']),mode='random', seed=args.seed)
        if psi == None:
            print ("Generating new psi.")
            
            L = model_par['Lx'] * model_par['Ly']
            upfilling, dnfilling = get_filling(run_par, L)
            prod = random_list(L, upfilling, dnfilling)
            psi = MPS.from_product_state(M.lat.mps_sites(), bc=model_par['bc_MPS'])#psi =  MPS.from_product_state(M.lat.mps_sites(), initial_state, dtype=complex, conserve=M, bc=model_par['bc'])
        else:
            print ("Using user-specified psi.")
        num_ini = psi.expectation_value(M.N)
        print ('Initial density profile:', num_ini)
    t0 = time.time()
    print ("Starting DMRG...")
    out = dmrg.DMRG.run(psi, M, sim_par)
    print ("DMRG took {} seconds.".format(time.time() - t0))
    out['time'] = time.time() - t0

		# sim = simulation.simulation(psi, M)

	# sim.dmrg_par = sim_par
	# sim.ground_state()
	# out = sim.sim_stats[-1]

    if out['shelve_flag']:  # DMRG did not end before time ran out; save sim to disk
        save_sim = {
			'LP': out['LP'],
			'RP': out['RP'],
			'psi': psi,
			'M': M,
			'num_ini': num_ini,
			'initial_state': initial_state,
			'model_par': model_par,
			'sim_par': sim_par,
		}
        with open('fh_dmrg_' + identifier + '.sim', 'wb') as f:
            pickle.dump(save_sim, f)

        out['skip'] = False

    return psi, out, num_ini, M, initial_state

def time_evolve(model_par, run_par, args):
# 	"""Initialize an MPS by time-evolving. Taken from evolve_a_block_hof_1.py.

# 	Args:
# 		model_par (dict): Model parameters
# 	    run_par (dict): Executable parameters
# 	    args (Namespace): All command line options

# 	Returns:
# 	    psi (MPS): The initialized state

# 	"""
# 	print ("Initializing for time evolution.")
# 	dt = run_par['dt']
# 	M = mod.HofstadterFermions(model_par)
# 	H = MPO.mpo_from_W(M.H_mpo, M.vL, M.vR, bc=M.bc) # mpo_from_W doesnt exist???
# 	#Us = [H.make_U(-1j*dt) ] #first order
# 	Us = [ H.make_U(-1j*dt*(1+1j)/2.), H.make_U(-1j*dt*(1-1j)/2.)] #2nd order
# 	initial_state = MPS.build_initial_state(int(args.Lx*model_par['flux_q']*model_par['Ly']), (0,1), (1-args.filling,args.filling), mode='random', seed=args.seed)
# 	psi = MPS.from_product_state(M.d, initial_state, dtype=complex, conserve=M, bc=model_par['bc']) #No d in M???
# 	truncation_par = {'chi_max': args.chi, 'trunc_cut': 1e-9, 'svd_max': 18.}  # Hardcoded but could be variable
# 	var_par = {'N_update_env': 1, 'max_iter': 1, 'min_iter': 1}  # Hardcoded but could be variable

# 	t = 0.
# 	step = 0
# 	print ("Now running time evolution"),
# 	while t < run_par['t_max']:
# 		print ("."),
# 		for s in range(run_par['evo_steps']):
# 			if step < M.L:
# 				for U in Us:
# 					U.apply_mpo(psi,compression='SVD', truncation_par=truncation_par)
# 			else:
# 				for U in Us:
# 					U.apply_mpo(psi,compression='VAR', truncation_par=truncation_par, var_par=var_par)
# 			step += 1
# 		t += run_par['evo_steps'] * dt

# 	if args.verbose > 0:
# 		dens = psi.expectation_value(M.N)
# 		print ('Density profile:'), dens
# 		print ('Total # particles:'), sum(dens)
# 		ent = psi.entanglement_entropy()
# 		print ('Entanglement entropy:'), ent
# 		print ('Average entanglement:'), np.mean(ent)
    pass
    return psi


def save_data(psi, M, args, initial_state, model_par, identifier, out, extra={}, save_state=False):
	# Check the particle density.
	num = psi.expectation_value(M.N)
	print (num)
	if args.plots:
		plt.plot(num, label='After DMRG')
		plt.plot(num_ini, label='Initial state')
		for i in range(len(initial_state)/model_par['Ly'] + 1):  # Vertical lines denoting rings of the cylinder.
			plt.axvline(x=(model_par['Ly'] * i - 0.5), color='r', ls='--')
		plt.xlabel('MPS index')
		plt.ylabel('Particle number')
		plt.title('fermi-hof <N>' + identifier)
		plt.legend()
		plt.savefig(identifier + '.png')
		try:
			plt.show()
		except:
			print ('plt.show() failed. Are you running remotely?')

	out = cleanup_output(out)
	data = {
		'num': num,
		'out': out,
		'model_par': model_par,
	}

	if args.compute_spec:  # Get entanglement spectra
		spec = psi.entanglement_spectrum()  # Gets spectrum for all bonds without specifying.
		spec_ring = spec[model_par['Ly'] - 1] # 15th bond cuts at a full ring when Ly = 8.
		lowest = {}  # Will only be filled if -plots is set.
		for q in spec_ring:
			charge, vals = q
			lowest[charge] = min(vals)
		# Get momentum-resolved entanglement spectrum
		perm = (1, model_par['Ly'])  # single site shift.
		k_spectrum = psi.compute_K(perm=perm, verbose=True, swapOp='f')  # swapOp 'f' for fermions
		print (k_spectrum)
		data.update({
			'spec': spec,
			'lowest': lowest,
			'k_spec': k_spectrum,
			'perm': perm,
		})
		identifier += '_spec'

	if args.compute_corr:  # Get correlation functions
		corr = psi.correlation_function(M.N, M.N, args.corr_cells * M.L, M.L)  # M.L gives us all correlations
		data['corr'] = corr
		identifier += '_corr'
		if args.corr_cells > 1:
			identifier += '_cells_{}'.format(args.corr_cells)

	if args.compute_cl:  # Get correlation length
		correlation_length = psi.correlation_length(verbose=args.verbose)  # get over all charge sectors.
		data['correlation_length'] = correlation_length
		identifier += '_clength'

	if save_state:
		data['psi'] = psi
		identifier += '_mps'


	########## Add any extra data (mostly useful for outside calls).
	if extra:
		print ("Extra data to save. Adding entries to data dict:")
		print (extra.keys())
		assert type(extra) == dict
		for extra_key in extra.keys():
			identifier += '_{}'.format(extra_key)
			assert extra_key not in data.keys()
		data.update(extra)
	else:
		print ("No extra data. Continue.")


	############## Save the data! ################
	with open('fh_dmrg_' + identifier + '.dat', 'wb') as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

	if args.plots:
		plt.figure()
		for q in spec_ring:
			# plt.plot(np.ones(len(q[1])) * q)
			charge, vals = q
			plt.plot(np.ones(len(vals)) * charge[0], vals, '_')
		print (lowest)

		try:
			plt.title('Entanglement spectrum for fermionic Hofstadter model')
			plt.xlabel('Charge label')
			plt.ylabel('Entanglement energy')
			plt.show()
		except:  # Don't want plotting to break the file.
			print ('Plotting failed. Are you running remotely?')

if __name__ == '__main__':
	run_defaults = {
		'N_MPS': 2,  # Nbr of particles in the system.
		'init_finite': False,
		'init_evo': False,
		't_max': 0.5,  # For time evolution
		'evo_steps': 1,  # For time evolution
		'dt': 0.05,  # For time evolution
		'compute_spec': False,  # Compute (momentum-resovled) entanglement spectrum
		'compute_corr': False,  # Compute ground state correlation functions
		'compute_cl': False,  # Compute correlation length
		'corr_cells': 1,  # Compute correlations to # number of unit cells
		'chi_scale': 0,  # If nonzero, perform chi-scaling with these steps. Save data at every point. Cannot be combined with initalizations.
		'chi_start': 0,  # Optional. If nonzero, start chi_scale from here.
		'save_state': False,  # Save the final state.
		'save_all': False,  # chi-scale only: save all states.
		'reload_sim': False,  # Reload simulation
		'start_from': 'none',  # Path to initial MPS file. Cannot be combined with initializations/chi-scaling.
		'min_steps': 20,
        'max_steps': 50,
		'nolanczos_sweeps': 1,
        
	}

	# Do the g.s. calculations (using above-defined functions)
	model_par, sim_par, run_par, args = misc.setup_executable(mod.HofstadterFermions, run_defaults)
	sim_par.update({
		'STARTING_ENV_FROM_PSI': args.nolanczos_sweeps,
		'MIN_STEPS': args.min_steps,
		'MAX_STEPS': args.max_steps,
	})
	run_par.update({
	    'filling': float(run_par['N_MPS']) / (model_par['Lx'] * model_par['Ly'] * model_par['flux_q']),
	    'N_MUC': float(run_par['N_MPS']) / (model_par['Lx'] * model_par['Ly']),
	    'band_filling': float(run_par['N_MPS']) / (model_par['Lx'] * model_par['Ly']),
	})
	identifier = run_par['identifier'] + '_N_MPS_{}'.format(run_par['N_MPS'])
	identifier += '_nolanczos_{}'.format(args.nolanczos_sweeps)
# 	if model_par['var_u'] == 0:
# 		identifier = identifier.replace('_u_shape_{}_'.format(model_par['u_shape']), '_')
# 	if model_par['pbc'] == 0:
# 		identifier += '_strip'
# 	elif model_par['pbc'] != 1:
# 		raise ValueError("pbc can only be 1 or 0.")
	if run_par['init_finite'] and run_par['init_evo']:
		raise ValueError("Cannot initialize as finite and with time evolution simultaneously!")
	if args.start_from != 'none':
		identifier += '_with_ini'
		sim_par['CHI_LIST'] = {0:args.chi}  # Assume we can start at max chi immediately
		sim_par['MIN_STEPS'] = 20
		with open(args.start_from, 'rb') as f:
				data = pickle.load(f)
				psi_start = data['psi']
	else:
		psi_start = None

	save_state = args.save_state
	sim_par['identifier'] = identifier

	if model_par['bc'] == 'periodic' and run_par['init_finite']:  # Iitialize as finite size
		print ('Initializing DMRG with finite bc.')
		model_par['bc'] = 'finite'
		psi, out, num_finite, M, initial_state = dmrg_wrapper(model_par, sim_par, run_par, args)
		print ('Now going to infinite.')
		psi.bc = 'periodic'
		model_par['bc'] = 'periodic'
		psi, out, num_ini, M, initial_state = dmrg_wrapper(model_par, sim_par,run_par, args, psi)
		if not out['skip']:
			save_data(psi, M, args, initial_state, model_par, identifier, out, save_state=args.save_state)

	elif run_par['init_evo']:  # Initialize with time evolution
		sim_par['CHI_LIST'] = {0:args.chi}
		psi = time_evolve(model_par, run_par, args)
		print ('psi has bond dimensions:', psi.chi)
		psi, out, num_ini, M, initial_state = dmrg_wrapper(model_par, sim_par, args, psi)
		if not out['skip']:
			save_data(psi, M, args, initial_state, model_par, identifier, out, save_state=args.save_state)

	elif run_par['chi_scale'] != 0:  # Do chi-scaling
		max_chi = max(sim_par['CHI_LIST'].values())
		old_chi = max_chi
		chi_start = run_par['chi_start'] if run_par['chi_start'] else run_par['chi_scale']
		print ("Begin chi-scaling. starting at", chi_start)
		for chi in range(chi_start, max_chi + 1, run_par['chi_scale']):
			print ("Chi-scaling now at chi =", chi)
			identifier = identifier.replace('chi_{}'.format(old_chi), 'chi_{}'.format(chi))
			sim_par['identifier'] = identifier  # Overwrite dict
			old_chi = chi
			if chi == chi_start:  # Initial run
				if save_state == 'none':
					sim_par['CHI_LIST'] = misc.chi_list(chi, args.dchi, args.dsweeps, args.verbose)
				else:
					sim_par['CHI_LIST'] = {0:chi}
				sim_par['MIN_STEPS'] = int(max(sim_par['CHI_LIST'].keys()) + 2 * args.dsweeps)
				psi, out, num_ini, M, initial_state = dmrg_wrapper(model_par, sim_par, run_par, args, psi=psi_start)
			else:  # Use lower-chi guess for next step. Immediately run at the new chi.
				sim_par['CHI_LIST'] = {0:chi}
				sim_par['MIN_STEPS'] = args.dsweeps
				psi, out, num_ini, M, initial_state = dmrg_wrapper(model_par, sim_par, run_par, args, psi)
			sim_par['shelve_after_time'] -= out.get('time', 0) / 3600  # Subtract runtime from shelve time.
			if not out['skip']:
				if chi != max_chi:
					save_data(psi, M, args, initial_state, model_par, identifier, out, save_state=args.save_all)
				else:  # Only save the final state (for the highest chi)
					save_data(psi, M, args, initial_state, model_par, identifier, out, save_state=(args.save_state or args.save_all))

	else:  # Default run
		sim_par['MIN_STEPS'] = args.min_steps  # Only works here
		psi, out, num_ini, M, initial_state = dmrg_wrapper(model_par, sim_par, run_par, args, psi=psi_start, sim_filename=identifier, reload_sim=args.reload_sim)
		if not out['skip']:  # Only save if we haven't yet; prevents endless overwriting
			save_data(psi, M, args, initial_state, model_par, identifier, out, save_state=args.save_state)

