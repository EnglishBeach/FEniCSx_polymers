import peewee as pe

database_connection = pe.MySQLDatabase(
    'orm_solves',
    user='root',
    password='',
    autoconnect=False,
)


def ready(cls=None, get=False, l=[]):
    if not get:
        l.append(cls)
        return cls
    return l


class BaseModel(pe.Model):

    class Meta:
        database = database_connection


@ready
class Mesh1D(BaseModel):

    class Meta:
        indexes = ((('left', 'right', 'domain_intervals','family', 'degree'), True), )

    left = pe.FloatField()
    right = pe.FloatField()
    intervals = pe.IntegerField()
    family = pe.CharField(10)
    degree = pe.IntegerField()
    description = pe.CharField(
        50,
        null=True,
    )


@ready
class Light1D(BaseModel):

    class Meta:
        indexes = ((('kind', 'left', 'right', 'slope'), True), )

    kind = pe.CharField(10)
    left = pe.FloatField()
    right = pe.FloatField()
    slope = pe.FloatField()


# FIXME: dicts
@ready
class Solver(BaseModel):
    petsc_opts = pe.CharField(100)
    solve_opts = pe.CharField(
        100,
        null=True,
    )
    form_opts = pe.CharField(
        100,
        null=True,
    )
    jit_opts = pe.CharField(
        100,
        null=True,
    )

    description = pe.CharField(
        50,
        null=True,
    )


@ready
class Task(BaseModel):
    name = pe.CharField(50)
    date = pe.DateTimeField(default=pe.datetime.datetime.now)
    description = pe.CharField(
        50,
        null=True,
    )

    T = pe.FloatField()
    dt = pe.FloatField()
    mesh = pe.ForeignKeyField(
        Mesh1D,
        backref='tasks',
        on_update='cascade',
    )
    N0 = pe.FloatField()
    P0 = pe.FloatField()
    light = pe.ForeignKeyField(
        Light1D,
        backref='tasks',
        on_update='cascade',
    )

    solve = pe.ForeignKeyField(
        Solver,
        backref='tasks',
        on_update='cascade',
    )


@ready
class Const(BaseModel):

    class Meta:
        indexes = ((('parametr', 'task'), True), )

    parametr = pe.CharField(20)
    value = pe.FloatField()
    task = pe.ForeignKeyField(
        Task,
        backref='consts',
        on_update='cascade',
    )


if __name__ == '__main__':
    with database_connection:
        database_connection.drop_tables(ready(get=True))
        database_connection.create_tables(ready(get=True))
