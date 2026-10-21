    ########################################################################
    #
    # State change matrix
    #
    ########################################################################

    # TODO: The folloiwng commented out matrices are not used
    #       consider removing if they are just providing clutter

    # def _computeReactantMatrix(self):
    #     """
    #     The reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is involved in transition j, \\\\
    #                                    0, &otherwise \\right.
    #     """
    #     # declare holder
    #     self._lambdaMat = np.zeros((self.num_state, self.num_transitions), int)

    #     _f, _t, eqn = self._unrollTransitionList(self._getAllTransition())
    #     for j, eqn in enumerate(eqn):
    #         for i, state in enumerate(self._stateList):
    #             if type(eqn)==int:
    #                 self._lambdaMat[i, j] = 0
    #             elif self._stateDict[state.ID] in eqn.atoms():
    #                 self._lambdaMat[i, j] = 1

    #     return self._lambdaMat

    # # Might replace _computeReactantMatrix. This function gives a matrix 
    # def _computeReactantMatrixOD(self):
    #     """
    #     The alternative reactant matrix, where

    #     .. math::
    #         \\lambda_{i,j} = \\left\\{ 1, &if state i is an origin or destination in transition j, \\\\
    #                                    0, &otherwise \\right.

    #     OD imples this refers to origin and destination
    #     """
        
    #     x=self._vMat!=0
    #     x=x.astype(int)
    #     self._lambdaMatOD=x

    #     return self._lambdaMatOD

    # def _computeDependencyMatrix(self):
    #     """
    #     Obtain the dependency matrix/graph. G_{i,j} indicate whether invoking
    #     the transition j will cause the rate to change for transition j
    #     """
    #     # if self._lambdaMat is None:
    #     #     self._computeReactantMatrix()
    #     # if self._lambdaMatOD is None:
    #     #     self._computeReactantMatrixOD()
    #     if self._vMat is None:
    #         self._computeStateChangeMatrix()

    #     nt = self.num_transitions
    #     self._GMat = np.zeros((nt, nt), int)

    #     for i in range(nt):
    #         for j in range(nt):
    #             d = 0
    #             for k in range(self.num_state):
    #                 d = d or (self._lambdaMat[k, i] and self._vMat[k, j])
    #             self._GMat[i, j] = d

    #     return self._GMat


    ########################################################################
    # Unrolling of the information
    # state
    # TODO: This unrolling is probably not useful anymore if we are
    #       basing the system on events rather than transitions.
    ########################################################################

    def _unrollState(self, state):
        """
        Information unrolling from vector to sympy in state
        """
        state_out = list()
        if self.num_state == 1:
            if isinstance(state, Number):
                state_out.append((self._state_store.symbol_list[0], state))
            else:
                raise InputError("Number of input state not as expected")
        else:
            if len(state) == self.num_state:
                for i, si in enumerate(self._state_store.symbol_list):
                    state_out.append((si, state[i]))
            else:
                raise InputError("Number of input state not as expected")

        return state_out

    def _unrollTransition(self, transition_obj):
        """
        Given a transition object, get the information from it in a usable
        format i.e. indexing within this class
        """
        from_index = self._extractStateIndex(transition_obj.origin)
        to_index = self._extractStateIndex(transition_obj.destination)
        eqn = checkEquation(transition_obj.equation, self)

        
        # Try returning as dict (should improve modularity over tuple output)

        out= {"from_index": from_index,
              "to_index": to_index,
              "eqn": eqn}

        return out

    def _unrollTransitionList(self, transition_list):
        '''
        ...describe...
        '''

        from_list = list()
        to_list = list()
        eqn_list = list()
        type_list = list()

        for transition_obj in transition_list:
            unrolled_transition=self._unrollTransition(transition_obj)
            from_list.append(unrolled_transition["from_index"])
            to_list.append(unrolled_transition["to_index"])
            eqn_list.append(unrolled_transition["eqn"])

        eqn_list = eqn_list if hasattr(eqn_list, '__iter__') else [eqn_list]

        out= {
            "from_list": from_list,
            "to_list": to_list,
            "eqn_list": eqn_list,
            }

        return out

    def _getAllTransition(self, pureTransitions=False):
        '''
        Get all transitions into a list
        If pureTransitions==True just transitions between states
        If pureTransitions==False between states plus birth deaths
        '''
        assert isinstance(pureTransitions, bool), "requires type(pureTransitions) = bool"

        if pureTransitions:
            return self._transitionList
        else:
            return self._transitionList+self._birthDeathList

    def _iterStateList(self):
        """
        Iterator through the states in symbolic form
        """
        for s in self._state_store.symbol_list:
            yield s

    def _iterParamList(self):
        """
        Iterator through the parameters in symbolic form
        """
        for p in self._parameter_store.symbol_list:
            yield p

    def _getListOfVariablesDict(self):
        # param_dict = [self._paramDict, self._stateDict, self._vectorStateDict]

        param_dict = [
            self._parameter_store.symbol_dict,
            self._state_store.symbol_dict,
            self._vectorStateDict
        ]

        return param_dict, self._derivedParamDict

    ########################################################################
    #
    # Ugly shit that is required to fix strings to sympy symbols
    #
    ########################################################################

    # def _extractParamIndex(self, input_str):
    #     if input_str in self._paramDict:
    #         return self._paramList.index(self._paramDict[input_str])
    #     else:
    #         raise InputError("Input parameter: %s does not exist" % input_str)

    # def _extractParamSymbol(self, input_str):
    #     """
    #     Given a parameter name, input_str
    #     """
    #     if isinstance(input_str, ODEVariable):
    #         input_str = input_str.ID

    #     if input_str in self._paramDict:
    #         return self._paramDict[input_str]
    #     else:
    #         raise InputError("Input parameter: %s does not exist" % input_str)

    # TODO: figure out why this is so awkward
    # def _extractStateIndex(self, input_str):
    #     '''
    #     Find the index of the string or sympy.Symbol 'input_str'
    #     '''
    #     if input_str is None:
    #         return list()
    #     else:
    #         if isinstance(input_str, (str, sympy.Symbol)):
    #             input_str = [input_str] # make this an iterable TODO: why?

    #         if hasattr(input_str, '__iter__'):
    #             return [self._extractStateIndexSingle(i) for i in input_str]
    #         else:
    #             raise Exception("Input must be a string or an iterable " +
    #                             "object of string")

    # def _extractStateIndexSingle(self, input_str):
    #     '''
    #     Find the index of the string or sympy.Symbol 'input_str'
    #     '''
    #     if isinstance(input_str, ODEVariable):
    #         return self._stateList.index(input_str)
    #     else:
    #         sym_name = self._extractStateSymbol(input_str)
    #         return self._stateList.index(sym_name)

    # def _extractStateSymbol(self, input_str):
    #     if isinstance(input_str, ODEVariable):
    #         input_str = input_str.ID

    #     if input_str in self._stateDict:
    #         return self._stateDict[input_str]
    #     else:
    #         sym_name = re_symbol_name.search(input_str)
    #         if sym_name is not None:
    #             if sym_name.group() in self._vectorStateDict:
    #                 index = re_symbol_index.findall(input_str)
    #                 if index is not None and len(index) == 1:
    #                     _i = int(index[0])
    #                     return self._vectorStateDict[sym_name.group()][_i]
    #                 else:
    #                     raise InputError("Cannot find input state, input {} " 
    #                                      "appears to be a vector that was " 
    #                                      "not initialized".format(sym_name))
    #             else:
    #                 raise InputError("Cannot find input state, input {} " 
    #                                  "likely to be a vector".format(sym_name))
    #         else:
    #             raise InputError("Input state: {} does not exist"
    #                              "".format(input_str))