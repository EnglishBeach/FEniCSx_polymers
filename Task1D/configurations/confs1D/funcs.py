import shutil as _shutil
import os as _os
from tqdm import tqdm as _tqdm
import jsonpickle as _jp
from dolfinx import io as _io
from fenics import operators as _fn
from .parametrs import Data


def SolveTimeDependent_1D(
    problem: _fn.NonlinearProblem,
    time_line,
    change_func,
    save_func,
    data: Data,
    domain,
    save=False,
):
    time_steps = _tqdm(
        desc=f'Solving PDE. Time:{0:.3f}',
        iterable=time_line,
    )
    if not save:
        for step in time_steps:
            time_steps.set_description(f'Solving PDE. Time:{step:.2f}')
            problem.solve()
            change_func()
    else:
        try:
            _shutil.rmtree(data.save_confs.dir + data.save_confs.name)
        except:
            pass
        if not _os.path.isdir(data.save_confs.dir + data.save_confs.name):
            _os.mkdir(data.save_confs.dir + data.save_confs.name)
        save_path = data.save_confs.dir + data.save_confs.name + '/' + data.save_confs.file_name

        with _io.XDMFFile(domain.comm, save_path + '.xdmf', 'w') as file:
            file.write_mesh(domain)

            for step in time_steps:
                if step in data.time.check:
                    time_steps.set_description(f'Solving PDE. Time:{step:.2f}')
                    save_func(file, step)

                problem.solve()
                change_func()

        elapsed = time_steps.format_dict['elapsed']
        total_time = time_steps.format_interval(elapsed)
        data({'solve status': 'solved', 'total time': total_time})
        with open(
                data.save_confs.dir + data.save_confs.name + '/annotation.txt',
                'w+') as file:
            file.write(str(data))

        with open(
                data.save_confs.dir + data.save_confs.name + '/dump.json',
                'w+') as file:
            file.write(_jp.encode(data,indent=1))

        with open(
                data.save_confs.dir + data.save_confs.name + '/dump_full.json',
                'w+') as file:
            file.write(_jp.encode(data.Options,indent=1))
