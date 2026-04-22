# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:27:33 2024

@author: REMOVED
"""

import json
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sshtunnel import SSHTunnelForwarder

import os

def connect_giantleaps_database():

    db_url = f'REMOVED'

    return create_engine(db_url)

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
    if db_type == 'PostgreSQL':
        if ssh == True:
            tunnel = SSHTunnelForwarder(
                ('REMOVED', 22),
                ssh_username='REMOVED',
                ssh_password='REMOVED',
                #ssh_private_key_password=secrets.ssh_private_key_password,
                remote_bind_address=('127.0.0.1', 5432)
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