using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 0);

// Carregando dados de um arquivo CSV ou outra fonte
IDataView dataView = mlContext.Data.LoadFromTextFile<AtividadeAluno>("C:\\Users\\vinic\\source\\repos\\ProjetoEduca\\Host\\melhorado_atividades_alunos.csv", hasHeader: true, separatorChar: ',');
var pipeline = mlContext.Transforms.Conversion.ConvertType("VaiBem", outputKind: DataKind.Boolean)
    .Append(mlContext.Transforms.Concatenate("Features", "AlunoId", "AtividadeId", "Pontuacao", "Duracao"))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "VaiBem", featureColumnName: "Features", maximumNumberOfIterations: 60));
var model = pipeline.Fit(dataView);
//var predictions = model.Transform(dataView);
//var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

var predictions = model.Transform(dataView);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "VaiBem");

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");


var predictionEngine = mlContext.Model.CreatePredictionEngine<AtividadeAluno, AtividadeAlunoPrediction>(model);

var novaAtividade = new AtividadeAluno {
    AlunoId = 1,
    AtividadeId = 121, // novo exercício
    Pontuacao = 99, // Pontuação do exercício anterior
    Duracao = 20 // Duração do exercício anterior
};

var previsao = predictionEngine.Predict(novaAtividade);
Console.WriteLine($"Previsão: {(previsao.PredictedVaiBem ? "Bom" : "Ruim")} com {previsao.Probability:P2} de confiança");

var classificacaoPrevista = predictionEngine.Predict(novaAtividade);
//Console.WriteLine($"Classificação: {classificacaoPrevista.PredictedClassificacao}");

//Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
//Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:F2}");
Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");


public class AtividadeAluno {
    [LoadColumn(0)]
    public float AlunoId { get; set; }

    [LoadColumn(1)]
    public float AtividadeId { get; set; }

    [LoadColumn(2)]
    public float Pontuacao { get; set; }

    [LoadColumn(3)]
    public float Duracao { get; set; }

    //[LoadColumn(4)]
    //public string Classificacao { get; set; } // Rótulo

    [LoadColumn(4)]
    public bool VaiBem { get; set; } // Note que é booleano
}

public class AtividadeAlunoPrediction {
    //[ColumnName("PredictedLabel")]

    //public string PredictedClassificacao { get; set; }
    [ColumnName("PredictedLabel")]
    public bool PredictedVaiBem { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}


