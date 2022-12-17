class Param_INTERFACE:

    def get_param(self, name):
        assert name != 'get_param'
        return getattr(self, name)


class Param_save(Param_INTERFACE):

    def __init__(self, save_name='test', desc=None,dir_save='/home/Solver1D/Solves/' ):
        self.dir_save = dir_save
        self.file_name = '/system1D'
        while True:
            if save_name == 'input':
                self.save_name = input('Set name')
            else:
                self.save_name = save_name
            if self.save_name != '': break

        if desc == 'input':
            self.desc = input('Set description')
        else:
            self.desc = desc


class Param_time(Param_INTERFACE):

    def __init__(self, T=1, n_steps=100, dt=1, n_shecks=10):
        if dt == 1:
            dt = T / n_steps
        elif n_steps == 100:
            n_steps = int(T / dt)
        elif T == 1:
            T = n_steps * dt
        assert T == n_steps * dt, ValueError('Incorrect time parametrs')
        self.T = T
        self.dt = dt
        self.n_steps = n_steps
        self.n_shecks = n_shecks


class Param_mesh(Param_INTERFACE):

    def __init__(
        self, left=0, right=1, domain_intervals=100, degree=1, family='CG'
    ):
        self.left = left
        self.right = right
        self.domain_intervals = domain_intervals
        self.family = family
        self.degree = degree


class Param_const(Param_INTERFACE):

    def __init__(
        self,
        gen_rate=0.01,
        P_step=0.13,
        a_rate=0.1,     # NM
        b_rate=1,     # PM
        e_rate=1     # NP
    ):
        self.gamma = 4
        self.P_step = P_step
        self.gen_rate = gen_rate
        self.a_rate = a_rate
        self.b_rate = b_rate
        self.e_rate = e_rate


class Param_light(Param_INTERFACE):

    def __init__(
        self,
        kind='sharp',
        left=0.4,
        right=0.6,
        slope=100,
    ):
        self.kind = kind
        self.left = left
        self.right = right
        self.slope = slope


class Param_initial(Param_INTERFACE):

    def __init__(self, N0=0.2, P0=0.001):
        self.N0 = N0
        self.P0 = P0


class Param_solve_confs(Param_INTERFACE):

    def __init__(
        self,
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        },
        solve_options={
            'convergence': 'incremental', 'tolerance': 1E-6
        },
        form_compiler_params={},
        jit_params={}
    ):
        self.petsc_options = petsc_options
        self.solve_options = solve_options
        self.form_compiler_params = form_compiler_params
        self.jit_params = jit_params


class Param_bcs(Param_INTERFACE):

    def __init__(self, bcs_kind='close', N_pars: list = {}, P_pars: list = {}):
        self.kind = bcs_kind
        self.N = N_pars
        self.P = P_pars


class Param_dump(Param_INTERFACE):

    def __init__(self, consts={}, equations={}):
        self.consts = consts
        self.equations = equations


class Param_DATA(Param_INTERFACE):

    def __init__(
        self,
        save=Param_save(),
        time=Param_time(),
        mesh=Param_mesh(),
        consts=Param_const(),
        light=Param_light(),
        initial=Param_initial(),
        bcs=Param_bcs(),
        solve_confs=Param_solve_confs(),
        dump=Param_dump()
    ):
        self.save = save
        self.time = time
        self.mesh = mesh
        self.consts = consts
        self.light = light
        self.initial = initial
        self.bcs = bcs
        self.solve_confs = solve_confs
        self.dump = dump
