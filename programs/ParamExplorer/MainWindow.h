#pragma once

#include "ExplorerModel.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMainWindow>
#include <QSlider>
#include <QTimer>
#include <QtCharts/QChartView>

#include <memory>
#include <vector>

// Qt5's Charts module puts everything in the QtCharts namespace (Qt6 dropped
// it). Without this, QChart resolves to QChar and the errors are unhelpful.
QT_CHARTS_USE_NAMESPACE

namespace explorer {

  /**
   * The explorer window: one slider per fit parameter, a stacked plot of the
   * selected sample projected onto the selected axis, and the current -2lnL.
   *
   * Evaluation is synchronous in the Qt event loop. The worst parameter costs
   * ~237 ms (see the design doc's Latency section), and the sliders do not track
   * during a drag, so one gesture triggers one evaluation on release rather than
   * dozens during the movement.
   */
  class MainWindow : public QMainWindow {
    Q_OBJECT

   public:
    explicit MainWindow(std::unique_ptr<ExplorerModel> model, QWidget* parent = nullptr);

   private:
    /** One parameter's row: the slider and the spin box are two views of one double. */
    struct Row {
      int             index = 0;
      QSlider*        slider = nullptr;
      QDoubleSpinBox* spin   = nullptr;
      double          lo     = 0.0;
      double          hi     = 0.0;
    };

    void build_ui();
    void rebuild_axis_choices();

    /** Evaluate, marginalize, redraw, update the status bar. The single refresh path. */
    void refresh();

    /** Ask for a refresh without doing one yet; the timer coalesces the requests. */
    void request_refresh() { m_Refresh->start(); }

    void set_from_slider(const Row& row, int tick);
    void set_from_spin(const Row& row, double value);

    std::unique_ptr<ExplorerModel> m_Model;

    QComboBox*  m_Sample = nullptr;
    QComboBox*  m_Axis   = nullptr;
    QCheckBox*  m_Split  = nullptr;
    QCheckBox*  m_LogY   = nullptr;
    QChartView* m_Chart  = nullptr;
    QLabel*     m_Status = nullptr;

    std::vector<Row> m_Rows;

    // Single-shot and restarted on every request, so a burst of changes (a
    // spin box arrowed several times, the sample and axis both switched)
    // collapses into one evaluation once the event queue drains.
    QTimer* m_Refresh = nullptr;
  };

}  // namespace explorer
