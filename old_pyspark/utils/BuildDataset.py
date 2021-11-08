from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def read_dataset(sqlContext):
    data = sqlContext.read.format("com.databricks.spark.csv").options(header='true',
                                                                      inferschema='true',
                                                                      delimiter='\t').load(
        'crc_msk_2017_clinical_data.tsv')
    # this will show data correlation and distribution, we don't need it all the time, it takes couple of min.
    # plot_data(data)

    return data


def plot_data(data):
    print(data.describe().toPandas().transpose())

    numeric_features = [t[0] for t in data.dtypes if t[1] == 'int' or t[1] == 'double']
    sampled_data = data.select(numeric_features).sample(False, 0.1).toPandas()
    axs = scatter_matrix(sampled_data, figsize=(10, 10))
    n = len(sampled_data.columns)
    for i in range(n):
        v = axs[i, 0]
        v.yaxis.label.set_rotation(0)
        v.yaxis.label.set_ha('right')
        v.set_yticks(())
        h = axs[n - 1, i]
        h.xaxis.label.set_rotation(90)
        h.set_xticks(())
    plt.savefig('../data_plot.png')

# print(clsf.test(lr_reg, test_line_df))
# testSchema = StructType([
#     StructField('patient_id', IntegerType(), False),
#     StructField('age', IntegerType(), False),
#     StructField('chemo_exposure', BooleanType(), False),
#     StructField('first_site_of_metastasis', ArrayType(), False),
#     StructField('fraction_genome_altered', FloatType(), False),
#     StructField('gene_panel', IntegerType(), False),
#     StructField('mCRC_type', IntegerType(), False),
#     StructField('metastasectomy', BooleanType(), False),
#     StructField('metastasectomy_site', IntegerType(), False),
#     StructField('metastases_site_first_bone', BooleanType(), False),
#     StructField('metastases_site_first_brain', BooleanType(), False),
#     StructField('metastases_site_first_gynecological', BooleanType(), False),
#     StructField('metastases_site_first_liver', BooleanType(), False),
#     StructField('metastases_site_first_ln', BooleanType(), False),
#     StructField('metastases_site_first_lung', BooleanType(), False),
#     StructField('metastases_site_first_pelvis', BooleanType(), False),
#     StructField('metastases_site_first_peritoneum_omentum_abdomen', BooleanType(), False),
#     StructField('metastatic_biopsy_site', IntegerType(), False),
#     StructField('msi_score', FloatType(), False),
#     StructField('msi_status', IntegerType(), False),
#     StructField('mutation_count', IntegerType(), False),
#     StructField('overall_survival_months', FloatType(), False),
#     StructField('living_status', BooleanType(), False),
#     StructField('other_metastasis_sites', BooleanType(), False),
#     StructField('patient_tumor_grade', IntegerType(), False),
#     StructField('primary_tumor_site', IntegerType(), False),
#     StructField('primary_tumor_location', IntegerType(), False),
#     StructField('sample_type', IntegerType(), False),
#     StructField('sex', IntegerType(), False),
#     StructField('specimen_type', IntegerType(), False),
#     StructField('stage_at_diagnostic', FloatType(), False),
#     StructField('time_from_met_dx_to_sequencing', FloatType(), False),
#     StructField('time_to_81_months', FloatType(), False),
# ])
# data = data.withColumn("first_site_of_metastasis", array(data["first_site_of_metastasis"]))
# data = data.withColumn("first_site_of_metastasis", data["first_site_of_metastasis"].cast("array<int>"))
# data = data.withColumn("metastasectomy_site", array(data["metastasectomy_site"]))
# data = data.withColumn("metastasectomy_site", data["metastasectomy_site"].cast("array<int>"))
