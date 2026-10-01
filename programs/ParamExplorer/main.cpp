#include "ExplorerModel.h"
#include "MainWindow.h"

#include <QApplication>
#include <QMessageBox>

#include <exception>
#include <memory>

int main(int argc, char** argv) {
  QApplication app(argc, argv);

  QStringList args = QApplication::arguments();
  // "--blind" mirrors LLHFit's own flag: signal parameters and bins above
  // 1e4 GeV get masked in the GUI, the same way --blind masks the writers'
  // output. The forward model underneath still uses every bin and parameter --
  // see ExplorerModel's constructor and ICBlinding.h.
  const bool blind = args.removeAll("--blind") > 0;

  if (args.size() < 2) {
    QMessageBox::critical(nullptr, "PhyLiNO Parameter Explorer",
                          QString("usage: %1 [--blind] <config.json>").arg(args.value(0, "PhyLiNOExplorer")));
    return EXIT_FAILURE;
  }

  try {
    // The parquet load takes 12-16 s on the combined config and happens before
    // any window exists, so the wait is unexplained unless it is announced.
    auto model = std::make_unique<explorer::ExplorerModel>(args[1].toStdString(), blind);

    auto* window = new explorer::MainWindow(std::move(model));
    window->show();

    return QApplication::exec();
  } catch (const std::exception& e) {
    QMessageBox::critical(nullptr, "PhyLiNO Parameter Explorer", e.what());
    return EXIT_FAILURE;
  }
}
