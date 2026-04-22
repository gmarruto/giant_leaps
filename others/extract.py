import pandas as pd
import json
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
from sshtunnel import SSHTunnelForwarder
import asyncio

def connect_database(db_conn_file, db_name, db_type, ssh = False):
    with open(db_conn_file) as f:
        db_conn_info = json.load(f)
###### for almaibar_ingurua  ##############
    if db_type == 'PostgreSQL':
        if ssh == True:
            tunnel = SSHTunnelForwarder(
                (db_conn_info['ssh']['host'], db_conn_info['ssh']['port']),
                ssh_username=db_conn_info['ssh']['user'],
                ssh_password=db_conn_info['ssh']['password'],
                #ssh_private_key_password=secrets.ssh_private_key_password,
                remote_bind_address=(db_conn_info[db_name]['host'], db_conn_info[db_name]['port'])
            )
            tunnel.start()
            port = tunnel.local_bind_port
            host = REMOVED

        else:
            port = db_conn_info[db_name]['port']
            host = db_conn_info[db_name]['host']

        db = db_conn_info[db_name]['db']
        pw = db_conn_info[db_name]['password']
        user = db_conn_info[db_name]['user']
        return create_async_engine("postgresql+asyncpg://{user}:{pw}@{host}:{port}/{db}"
                             .format(user=user,
                                     pw=pw,
                                     db=db,
                                     port=port,
                                     host=host))

###### for db-azti-nutrisport #############
    elif db_type == 'SQLServer':
        server_name = db_conn_info[db_name]['host']
        port = db_conn_info[db_name]['port']
        db = db_conn_info[db_name]['db']
        user = db_conn_info[db_name]['user']
        pw = db_conn_info[db_name]['password']

        pyodbc_connection_string = (
            f'Driver={{SQL Server}};'
            f'Server={server_name},{port};'
            f'Database={db};'
            f'PWD={pw};'
            f'UID={user};'
            'Trusted_Connection=No;Encrypt=Yes;'
        )

        return create_async_engine(f"mssql+asyncpg:///?odbc_connect={pyodbc_connection_string}")

    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def connect_sync_database(db_conn_file, db_name, db_type, ssh = False):
    with open(db_conn_file) as f:
        db_conn_info = json.load(f)
###### for almaibar_ingurua  ##############
    print(db_conn_info[db_name])
    if db_type == 'PostgreSQL':
        if ssh == True:
            # tunnel = SSHTunnelForwarder(
            #     (db_conn_info['ssh']['host'], db_conn_info['ssh']['port']),
            #     ssh_username=db_conn_info['ssh']['user'],
            #     ssh_password=db_conn_info['ssh']['password'],
            #     #ssh_private_key_password=secrets.ssh_private_key_password,
            #     remote_bind_address=(db_conn_info[db_name]['host'], db_conn_info[db_name]['port'])
            # )

            tunnel = SSHTunnelForwarder(
            #('172.17.146.40', 22),
            ('REMOVED',22),
            ssh_username="REMOVED",
            ssh_password="REMOVED",
            remote_bind_address=(REMOVED, REMOVED)
                )

            tunnel.start()
            print(tunnel.local_bind_port)
            port = tunnel.local_bind_port
            host = REMOVED

        else:
            port = db_conn_info[db_name]['port']
            host = db_conn_info[db_name]['host']

        print('SSh Connection Established')

        db = db_conn_info[db_name]['db']
        pw = db_conn_info[db_name]['password']
        user = db_conn_info[db_name]['user']
        return create_engine("postgresql://{user}:{pw}@{host}:{port}/{db}"
                             .format(user=user,
                                     pw=pw,
                                     db=db,
                                     port=port,
                                     host=host))

###### for db-azti-nutrisport #############
    elif db_type == 'SQLServer':
        server_name = db_conn_info[db_name]['host']
        port = db_conn_info[db_name]['port']
        db = db_conn_info[db_name]['db']
        user = db_conn_info[db_name]['user']
        pw = db_conn_info[db_name]['password']

        pyodbc_connection_string = (
            f'Driver={{SQL Server}};'
            f'Server={server_name},{port};'
            f'Database={db};'
            f'PWD={pw};'
            f'UID={user};'
            'Trusted_Connection=No;Encrypt=Yes;'
        )

        return create_engine(f"mssql:///?odbc_connect={pyodbc_connection_string}")

    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    
def ssh_tunnel():
    tunnel = SSHTunnelForwarder(
    #('172.17.146.40', 22),
    ('REMOVED',22),
    ssh_username="REMOVED",
    ssh_password="REMOVED",
    remote_bind_address=(REMOVED, REMOVED)
        )

    tunnel.start()

    print(tunnel.local_bind_port)  # show assigned local port
    # work with `SECRET SERVICE` through `server.local_bind_port`.

async def extract_food_sources(engine, target_file):

    async with engine.connect() as conn:
        select_protein_source_query = 'SELECT\
                            *\
                        FROM\
                            dcf_data.food_sources fs2;'

        trans = await conn.begin()
        df_foodsources_in_db = (await conn.run_sync(
                    lambda sync_conn: pd.read_sql_query(text(select_protein_source_query), con=sync_conn)
                ))
        await trans.commit()
    
    df_foodsources_in_db.to_excel(target_file)

async def extract_food_sources(engine, target_file):
    print('Extract food sources')
    async with engine.connect() as conn:
        select_protein_source_query = 'SELECT\
                            *\
                        FROM\
                            dcf_data.food_sources fs2;'

        trans = await conn.begin()
        df_foodsources_in_db = (await conn.run_sync(
                    lambda sync_conn: pd.read_sql_query(text(select_protein_source_query), con=sync_conn)
                ))
        await trans.commit()
    
    df_foodsources_in_db.to_excel(target_file)
    print('saved')

async def extract_food_source_relations(engine, target_file):
    async with engine.connect() as conn:
        select_relations_query = 'SELECT\
                                            *\
                                        FROM\
                                            dcf_data.food_data_source_relations fdsr;'
        #select_nutrients_query = 'select * from dcf_data.nutrients'
        trans = await conn.begin()
        df_relations_in_db = (await conn.run_sync(
                    lambda sync_conn: pd.read_sql_query(text(select_relations_query), con=sync_conn)
                ))
        await trans.commit()
    df_relations_in_db.to_excel(target_file)

async def extract_nutrient_composition_facts(engine, target_file):
    print('Extract nutrient composition facts')
    df_nutrient_facts_in_db_delta = pd.DataFrame()
    df_nutrient_facts_in_db_delta.to_excel(target_file)
    limit = 1000
    offset = 0
    empty_delta = True
    while empty_delta:
        async with engine.connect() as conn:
            select_nutrient_facts_query = 'SELECT\
                                    *\
                                FROM\
                                    dcf_data.nutrient_composition_facts ncf\
                                LIMIT 1000 OFFSET {};'.format(offset)
            #select_nutrients_query = 'select * from dcf_data.nutrients'
            trans = await conn.begin()
            df_nutrient_facts_in_db_delta = (await conn.run_sync(
                        lambda sync_conn: pd.read_sql_query(text(select_nutrient_facts_query), con=sync_conn)
                    ))
            await trans.commit()
            if len(df_nutrient_facts_in_db_delta) == 0:
                empty_delta = False
            else:
                with pd.ExcelWriter(target_file, mode='a', if_sheet_exists="overlay") as writer:
                    df_nutrient_facts_in_db_delta.to_excel(writer, sheet_name='Sheet1', startrow = offset)
                offset = offset + limit
                print(offset)
    print('saved')

async def extract_nutrients(engine, target_file):
    async with engine.connect() as conn:
        select_nutrients_query = 'SELECT\
                                        *\
                                    FROM\
                                        dcf_data.nutrients n;'
        #select_nutrients_query = 'select * from dcf_data.nutrients'
        trans = await conn.begin()
        df_nutrients_in_db = (await conn.run_sync(
                    lambda sync_conn: pd.read_sql_query(text(select_nutrients_query), con=sync_conn)
                ))
        await trans.commit()
    df_nutrients_in_db.to_excel(target_file)

async def main_extraction():
    db_connection_file = '../others/db_credentials.json'
    gl_engine = connect_database(db_connection_file, REMOVED, 'PostgreSQL')

    await extract_food_sources(gl_engine, '../datasets/food_sources.xlsx')
    # await extract_nutrients(gl_engine, '../datasets/nutrients.xlsx')
    # await extract_food_source_relations(gl_engine, '../datasets/food_source_relations.xlsx')
    # await extract_nutrient_composition_facts(gl_engine, '../datasets/nutrient_composition_facts.xlsx')


def test_connectivity():
    db_connection_file = '../others/db_credentials.json'
    gl_engine = connect_sync_database(db_connection_file, REMOVED, 'PostgreSQL', ssh = True)
    print('PostgreSQL connection established')
    print(gl_engine)
    print(gl_engine.connect())

    select_nutrients_query = 'SELECT\
                                        *\
                                    FROM\
                                        dcf_data.nutrients n;'
    df_nutrients_in_db = pd.read_sql_query(text(select_nutrients_query), con=gl_engine)
    print(df_nutrients_in_db)

    print('y aquí no entra')


    with gl_engine.connect() as conn:
        select_nutrients_query = 'SELECT\
                                        *\
                                    FROM\
                                        dcf_data.nutrients n;'
        trans = conn.begin()
        print('Empieza trans')
        df_nutrients_in_db = pd.read_sql_query(text(select_nutrients_query), con=gl_engine)
        trans.commit()
        print('Acaba trans')
        print(df_nutrients_in_db)


if __name__ == "__main__":
    import asyncio

    import ast

    #asyncio.run(test_connectivity())

    db_url = f'REMOVED'

    # print(db_url)
    # engine = create_engine(db_url)

    # select_nutrients_query = 'SELECT\
    #                                     *\
    #                                 FROM\
    #                                     dcf_data.nutrients n;'

    # # Ejecutar una consulta de prueba
    # with engine.connect() as connection:
    #     result = connection.execute(text(select_nutrients_query))
    #     for row in result:
    #         print(row)

    #test_connectivity()
    
    ssh_host = 'REMOVED'
    ssh_port = 22
    ssh_user = 'REMOVED'
    ssh_password = 'REMOVED'  # o usa clave privada

    db_host = REMOVED
    db_port = REMOVED
    db_user = REMOVED
    db_password = REMOVED
    db_name = REMOVED

    # Crear túnel SSH
    with SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_password,  # o usa `ssh_pkey='ruta_a_clave_privada'`
        remote_bind_address=(db_host, db_port)
    ) as tunnel:
        local_port = tunnel.local_bind_port
        
    # Crear la URL de conexión para SQLAlchemy
    db_url = f'postgresql://{db_user}:{db_password}@127.0.0.1:{local_port}/{db_name}'
        
    print(db_url)
    engine = create_engine(db_url)

        
    #     # Ejecutar una consulta de prueba
    with engine.connect() as connection:
        result = connection.execute("SELECT version();")
        for row in result:
            print(row)


