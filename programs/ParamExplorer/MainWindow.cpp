#include "MainWindow.h"

#include <QGridLayout>
#include <QGroupBox>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QLogValueAxis>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>

#include <QPainter>

#include <algorithm>
#include <cmath>
#include <iterator>

QT_CHARTS_USE_NAMESPACE

namespace explorer {

  namespace {

    // Sliders are integers; the model's value is a double. The tick count sets
    // the resolution of a drag, not the precision of the value -- the spin box
    // writes the double straight through.
    constexpr int kTicks = 1000;

    // QLogValueAxis cannot place a point at or below zero, and an empty bin is a
    // perfectly ordinary thing for a prediction to contain. Those bins are drawn
    // at the bottom of the axis instead of dropping out of the polyline.
    constexpr double kLogFloor = 1.0e-3;

    /**
     * Stack colours, in the draw order of ExplorerModel's component list. Chosen
     * to stay distinguishable when filled and overlapping, and to keep the
     * atmospheric halves visibly related to each other when the split is on.
     */
    const QColor kComponentColors[] = {
      QColor(0x1f, 0x77, 0xb4),  // astro
      QColor(0xff, 0x7f, 0x0e),  // atmospheric (or its conventional half)
      QColor(0xff, 0xbb, 0x78),  // atmospheric prompt
      QColor(0x2c, 0xa0, 0x2c),  // template
      QColor(0x94, 0x67, 0xbd),  // galactic
    };

    /** Turn per-bin values into a step polyline over the bin edges. */
    QLineSeries* step_series(const std::vector<double>& edges, const std::vector<double>& values,
                             bool log_y) {
      auto* series = new QLineSeries();
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double y = log_y ? std::max(values[i], kLogFloor) : values[i];
        series->append(edges[i], y);
        series->append(edges[i + 1], y);
      }

      return series;
    }

  }  // namespace

  MainWindow::MainWindow(std::unique_ptr<ExplorerModel> model, QWidget* parent)
    : QMainWindow(parent)
    , m_Model(std::move(model)) {
    build_ui();
    rebuild_axis_choices();
    refresh();
  }

  void MainWindow::build_ui() {
    setWindowTitle("PhyLiNO Parameter Explorer");

    auto* toolbar = addToolBar("View");

    m_Sample = new QComboBox();
    for (const std::string& name : m_Model->sample_names())
      m_Sample->addItem(QString::fromStdString(name));

    m_Axis = new QComboBox();

    m_Split = new QCheckBox("Split atmospheric");
    m_Split->setToolTip(
        "Draw the conventional and prompt halves separately.\n"
        "This re-walks every MC event, so it is much slower than the summed entry.");

    m_LogY = new QCheckBox("Log counts");
    m_LogY->setChecked(true);

    toolbar->addWidget(new QLabel(" Sample "));
    toolbar->addWidget(m_Sample);
    toolbar->addWidget(new QLabel(" Axis "));
    toolbar->addWidget(m_Axis);
    toolbar->addSeparator();
    toolbar->addWidget(m_Split);
    toolbar->addWidget(m_LogY);

    m_Chart = new QChartView();
    m_Chart->setRenderHint(QPainter::Antialiasing);
    m_Chart->setMinimumHeight(300);

    // One row per parameter, fixed ones included but disabled: hiding them would
    // reshuffle the panel between configs, and their values are still worth
    // reading.
    auto* grid = new QGridLayout();
    int   row  = 0;
    for (const ParamInfo& info : m_Model->parameters()) {
      Row r{.index = info.index, .lo = info.lo, .hi = info.hi};

      auto* label = new QLabel(QString::fromStdString(info.name));

      r.slider = new QSlider(Qt::Horizontal);
      r.slider->setRange(0, kTicks);
      r.slider->setValue(static_cast<int>(std::lround((info.value - r.lo) / (r.hi - r.lo) * kTicks)));
      // The evaluation is too slow for a live drag on the worst parameters, and
      // inconsistent tracking between rows would be worse than none.
      r.slider->setTracking(false);

      r.spin = new QDoubleSpinBox();
      // Deliberately wider than the slider: typing a value outside the slider's
      // range is a legitimate thing to want, and it simply pins the slider to an
      // end rather than being clamped away.
      r.spin->setRange(r.lo - 1000.0 * std::abs(r.hi - r.lo), r.hi + 1000.0 * std::abs(r.hi - r.lo));
      r.spin->setDecimals(4);
      r.spin->setSingleStep((r.hi - r.lo) / 100.0);
      r.spin->setValue(info.value);

      if (info.fixed) {
        label->setEnabled(false);
        r.slider->setEnabled(false);
        r.spin->setEnabled(false);
        label->setToolTip("Fixed in the config");
      }

      grid->addWidget(label, row, 0);
      grid->addWidget(r.slider, row, 1);
      grid->addWidget(r.spin, row, 2);
      ++row;

      m_Rows.push_back(r);
    }

    // Captured by value after the vector is done growing: a reference into
    // m_Rows would dangle as it reallocated.
    for (const Row& r : m_Rows) {
      connect(r.slider, &QSlider::valueChanged, this, [this, r](int tick) { set_from_slider(r, tick); });
      connect(r.spin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
              [this, r](double value) { set_from_spin(r, value); });
    }

    auto* panel = new QWidget();
    panel->setLayout(grid);

    auto* scroll = new QScrollArea();
    scroll->setWidget(panel);
    scroll->setWidgetResizable(true);

    auto* splitter = new QSplitter(Qt::Vertical);
    splitter->addWidget(m_Chart);
    splitter->addWidget(scroll);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);

    setCentralWidget(splitter);

    m_Status = new QLabel();
    statusBar()->addWidget(m_Status);

    m_Refresh = new QTimer(this);
    m_Refresh->setSingleShot(true);
    m_Refresh->setInterval(0);
    connect(m_Refresh, &QTimer::timeout, this, &MainWindow::refresh);

    connect(m_Sample, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
      rebuild_axis_choices();
      request_refresh();
    });
    connect(m_Axis, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { request_refresh(); });
    connect(m_Split, &QCheckBox::toggled, this, [this](bool) { request_refresh(); });
    connect(m_LogY, &QCheckBox::toggled, this, [this](bool) { request_refresh(); });

    resize(1100, 850);
  }

  void MainWindow::rebuild_axis_choices() {
    const QSignalBlocker blocker(m_Axis);

    m_Axis->clear();
    for (const AxisInfo& axis : m_Model->axes(static_cast<std::size_t>(m_Sample->currentIndex())))
      m_Axis->addItem(QString::fromStdString(axis.kind_name));
  }

  void MainWindow::set_from_slider(const Row& row, int tick) {
    const double value = row.lo + (row.hi - row.lo) * tick / kTicks;

    const QSignalBlocker blocker(row.spin);
    row.spin->setValue(value);

    m_Model->set(row.index, value);
    request_refresh();
  }

  void MainWindow::set_from_spin(const Row& row, double value) {
    // Pinned rather than clamped: the model takes the typed value even when the
    // slider cannot represent it, so the plot shows what was asked for.
    const int tick = std::clamp(static_cast<int>(std::lround((value - row.lo) / (row.hi - row.lo) * kTicks)),
                                0, kTicks);

    const QSignalBlocker blocker(row.slider);
    row.slider->setValue(tick);

    m_Model->set(row.index, value);
    request_refresh();
  }

  void MainWindow::refresh() {
    const auto sample = static_cast<std::size_t>(std::max(0, m_Sample->currentIndex()));
    const auto axis   = static_cast<std::size_t>(std::max(0, m_Axis->currentIndex()));

    const double llh = m_Model->evaluate();
    const auto   marginalized = m_Model->marginalize(sample, axis, m_Split->isChecked());

    const bool log_y = m_LogY->isChecked();

    auto* chart = new QChart();
    chart->setTitle(QString("%1 -- %2")
                        .arg(m_Sample->currentText())
                        .arg(m_Axis->currentText()));

    // Cumulative upper edges: each component's area sits on top of the ones
    // before it, so the top of the stack is the total prediction.
    std::vector<double> cumulative(marginalized.data.size(), 0.0);
    QLineSeries*        lower = step_series(marginalized.edges, cumulative, log_y);
    lower->setVisible(false);
    chart->addSeries(lower);

    double max_y = 0.0;

    for (std::size_t c = 0; c < marginalized.components.size(); ++c) {
      const auto& component = marginalized.components[c];
      for (std::size_t b = 0; b < cumulative.size(); ++b)
        cumulative[b] += component.values[b];

      QLineSeries* upper = step_series(marginalized.edges, cumulative, log_y);

      auto* area = new QAreaSeries(upper, lower);
      area->setName(QString::fromStdString(component.name));
      const QColor color = kComponentColors[std::min(c, std::size(kComponentColors) - 1)];
      area->setColor(color);
      area->setBorderColor(color.darker(130));
      chart->addSeries(area);

      lower = upper;
      max_y = std::max(max_y, *std::max_element(cumulative.begin(), cumulative.end()));
    }

    // The data on top of the stack. Under UseData: false this is the Asimov
    // expectation, so at the config's start point the points land exactly on the
    // top of the stack -- which is the cheapest available check that the
    // marginalization is right.
    auto* data = new QScatterSeries();
    data->setName("data");
    data->setMarkerSize(6.0);
    data->setColor(Qt::black);
    for (std::size_t b = 0; b < marginalized.data.size(); ++b) {
      const double centre = 0.5 * (marginalized.edges[b] + marginalized.edges[b + 1]);
      const double y      = log_y ? std::max(marginalized.data[b], kLogFloor) : marginalized.data[b];
      data->append(centre, y);
      max_y = std::max(max_y, marginalized.data[b]);
    }
    chart->addSeries(data);

    auto* x_axis = new QValueAxis();
    x_axis->setTitleText(m_Axis->currentText());
    x_axis->setRange(marginalized.edges.front(), marginalized.edges.back());
    chart->addAxis(x_axis, Qt::AlignBottom);

    QAbstractAxis* y_axis = nullptr;
    if (log_y) {
      auto* log_axis = new QLogValueAxis();
      log_axis->setBase(10.0);
      log_axis->setRange(kLogFloor, std::max(max_y * 1.5, 10.0 * kLogFloor));
      y_axis = log_axis;
    } else {
      auto* linear = new QValueAxis();
      linear->setRange(0.0, std::max(max_y * 1.1, 1.0));
      y_axis = linear;
    }
    y_axis->setTitleText("events");
    chart->addAxis(y_axis, Qt::AlignLeft);

    for (QAbstractSeries* series : chart->series()) {
      series->attachAxis(x_axis);
      series->attachAxis(y_axis);
    }

    // The hidden baseline of the bottom area would otherwise take a legend slot
    // of its own.
    chart->legend()->markers(lower).clear();
    chart->legend()->setAlignment(Qt::AlignRight);

    // setChart() takes ownership and deletes the previous chart with all its
    // series, so the ones built above need no separate cleanup.
    m_Chart->setChart(chart);

    m_Status->setText(QString("-2lnL = %1     delta = %2")
                          .arg(llh, 0, 'f', 4)
                          .arg(llh - m_Model->reference_likelihood(), 0, 'f', 4));
  }

}  // namespace explorer
