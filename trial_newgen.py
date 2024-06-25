    def montecarlostep_gen2(self):
        '''Propagating step. Uses a Metropolis algorithm.
        Each time method is called, entire system updated! New configuration then accepted or rejected.
        '''
        # random moves
        # choose random dna index, in same fashion as interactivity, segment and vector
        dnastr_A_new,dnastr_B_new = self.try_change(index_rand)

        # test random change against simulation requirements; intact, no overlap, confinement
        while self.check_excvol(dnastr_A_new[index_rand])* self.check_strintact(dnastr_A_new,index_rand)* self.check_inbox(dnastr_new[index_rand]) == 0: #need to slightly rewrite some of these to include both new A&B
 self.try_change(index_rand)

energy_old = self.eng()
energy_new = self.eng_elastic(dnastr_new,dnastr_other) + self.eng_elec(dnastr_new, dnastr_other)
delt_eng = energy_new - energy_old
            # could use the index and its neighbours to calculate the energy change directly

if delt_eng <= 0: # assign proposed string change
         self.dnastr_A, self.dnastr_B = dnastr_A_new, dnastr_B_new
           self.trajectories.append([self.dnastr_A,self.dnastr_B])

elif delt_eng >= 0:
         random_factor = np.random.random()
         boltzmann_factor = np.e**(-1*delt_eng/(kb*temp)) # delt_eng in kb units
         if random_factor < boltzmann_factor: # assign proposed string change
         self.dnastr_A, self.dnastr_B = dnastr_A_new, dnastr_B_new

                          self.trajectories.append([self.dnastr_A,self.dnastr_B])

        self.n_steps += 1


def try_change(self, index_rand):
    
vector_list_&0 = vector_list + [0]
        vector_rand = vector_list_&0[np.random.randint(7)]

        dnastr_A_new = self.dnastr_A[:index_rand] + [self.dnastr_A[index_rand]+vector_rand] + self.dnastr_A[index_rand+1:]

dnastr_B_new = self.dnastr_B[:index_rand] + [self.dnastr_B[index_rand]+vector_rand] + self.dnastr_B[index_rand+1:]

return dnastr_A_new, dnastr_B_new