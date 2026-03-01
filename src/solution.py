"""Stub da solução do case: cálculo mensal e rolling 3 meses por escola."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F

from src.utils import validate_payments_schema


def compute_overdue_rolling_3m(payments_df: DataFrame) -> DataFrame:
    """Computa inadimplência mensal e rolling 3M por escola.

    Args:
        payments_df: DataFrame de entrada com schema `payments`.

    Returns:
        DataFrame por `school_id` e `month` com colunas:
            - school_id
            - month
            - year_month
            - total_due_amount_month
            - total_overdue_amount_month
            - overdue_ratio_month
            - overdue_ratio_rolling_3m

    Tratamento de meses faltantes:
      - Internamente criei uma série mensal completa por escola
        para que o rolling 3M considere meses calendário reais.
      - Meses ausentes são preenchidos com 0.
      - No final, retornei apenas os meses existentes no input
        (para respeitar o contrato do teste).
    """

    validate_payments_schema(payments_df)

    # 1) Cria a chave mensal e guarda as combinações originais usadas depois para garantir que não adicionei linhas novas
    
    base = payments_df.withColumn("month", F.trunc(F.col("due_date"), "month"))
    orig_keys = base.select("school_id", "month").distinct()

    # 2) Agregação mensal por escola
    
    monthly = (
        base.groupBy("school_id", "month")
        .agg(
            # Soma total do mês
            F.sum(F.col("amount")).alias("total_due_amount_month"),

            # Soma apenas valores em atraso/abertos
            F.sum(
                F.when(F.col("status").isin("late", "open"), F.col("amount"))
                 .otherwise(F.lit(0))
            ).alias("total_overdue_amount_month"),
        )
    )

    
    # 3) Densificação mensal por escola (cria todos os meses entre min e max observados)
    
    school_ranges = monthly.groupBy("school_id").agg(
        F.min("month").alias("min_month"),
        F.max("month").alias("max_month"),
    )

    # Gera sequência mensal por escola
    school_calendar = (
        school_ranges.select(
            "school_id",
            F.explode(
                F.sequence(
                    F.col("min_month"),
                    F.col("max_month"),
                    F.expr("interval 1 month")
                )
            ).alias("month"),
        )
    )

    # Junta calendário com dados reais e preenche meses faltantes com 0
    monthly_full = (
        school_calendar
        .join(monthly, ["school_id", "month"], "left")
        .fillna({
            "total_due_amount_month": 0,
            "total_overdue_amount_month": 0,
        })
        .withColumn("year_month", F.date_format(F.col("month"), "yyyy-MM"))
    )

    # 4) Cálculo da razão mensal (com proteção contra divisão por zero)
    
    monthly_full = monthly_full.withColumn(
        "overdue_ratio_month",
        F.when(
            F.col("total_due_amount_month") > 0,
            F.col("total_overdue_amount_month") / F.col("total_due_amount_month"),
        ).otherwise(F.lit(0.0)),
    )

    # 5) Rolling 3 meses calendário. Como a série está densificada, rowsBetween(-2,0) equivale a mês atual + 2 anteriores reais
    
    w3 = (
        Window
        .partitionBy("school_id")
        .orderBy(F.col("month"))
        .rowsBetween(-2, 0)
    )

    monthly_full = (
        monthly_full
        .withColumn(
            "total_due_amount_rolling_3m",
            F.sum("total_due_amount_month").over(w3)
        )
        .withColumn(
            "total_overdue_amount_rolling_3m",
            F.sum("total_overdue_amount_month").over(w3)
        )
        .withColumn(
            "overdue_ratio_rolling_3m",
            F.when(
                F.col("total_due_amount_rolling_3m") > 0,
                F.col("total_overdue_amount_rolling_3m")
                / F.col("total_due_amount_rolling_3m"),
            ).otherwise(F.lit(0.0)),
        )
    )

    # 6) Retorna apenas meses presentes no input original -> garante que o resultado respeita o contrato do teste
    
    result = monthly_full.join(orig_keys, ["school_id", "month"], "inner")

    return result.select(
        "school_id",
        "month",
        "year_month",
        "total_due_amount_month",
        "total_overdue_amount_month",
        "overdue_ratio_month",
        "overdue_ratio_rolling_3m",
    )