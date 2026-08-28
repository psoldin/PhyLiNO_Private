#include "MainWindow.h"

#include <QGridLayout>
#include <QGroupBox>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBar>
#include <QVBoxLayout>
#include <QtCharts/QChart>
#include <QtCharts/QLineSeries>
#include <QtCharts/QLogValueAxis>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>

#include <QPainter>
#include <QPen>

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

    /**
     * Component colours, keyed by name rather than by draw order so that
     * atmospheric_conv and atmospheric_prompt keep their own identities when the
     * split is toggled and the list length changes.
     *
     * conv and prompt are deliberately far apart. They were near-identical
     * oranges at first, which made a prompt curve three decades below the
     * conventional one hard to pick out even once it was drawn as its own line.
     */
    QColor component_color(const std::string& name) {
      if (name == "astro")               return QColor(0x1f, 0x77, 0xb4);  // blue
      if (name == "atmospheric" ||
          name == "atmospheric_veto")    return QColor(0xff, 0x7f, 0x0e);  // orange
      if (name == "atmospheric_conv")    return QColor(0xff, 0x7f, 0x0e);  // orange
      if (name == "atmospheric_prompt")  return QColor(0xd6, 0x27, 0x28);  // red
      if (name == "template")            return QColor(0x2c, 0xa0, 0x2c);  // green
      if (name == "galactic")            return QColor(0x94, 0x67, 0xbd);  // purple
      return QColor(0x7f, 0x7f, 0x7f);
    }

    /** Where a value sits on a 0..kTicks slider spanning [lo, hi]. */
    int tick_of(double lo, double hi, double value) {
      return std::clamp(static_cast<int>(std::lround((value - lo) / (hi - lo) * kTicks)), 0, kTicks);
    }

    /** Turn per-bin values into a step polyline over the bin edges. */
    QLineSeries* step_series(const std::vector<double>& edges, const std::vector<double>& values,
                             double floor) {
      auto* series = new QLineSeries();
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double y = std::max(values[i], floor);
        series->append(edges[i], y);
        series->append(edges[i + 1], y);
      }

      return series;
    }

    /** Largest value across everything that will be drawn. */
    double max_of(const Marginalized& m) {
      double max_y = 0.0;
      for (const double v : m.total)
        max_y = std::max(max_y, v);
      for (const double v : m.data)
        max_y = std::max(max_y, v);
      for (const auto& component : m.components)
        for (const double v : component.values)
          max_y = std::max(max_y, v);

      return max_y;
    }

    /**
     * Bottom of a log axis: five decades below the peak, which keeps a component
     * that is a fraction of a percent of the total on the plot without letting a
     * single empty bin stretch the axis down to nothing. A fixed floor cannot do
     * both -- the samples here span 26 000 events and 0.06.
     */
    double log_floor(double max_y) {
      return max_y > 0.0 ? max_y * 1.0e-5 : 1.0e-3;
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

    auto* reset = new QPushButton("Reset");
    reset->setToolTip("Put every parameter back to its StartValue from the config.");

    toolbar->addWidget(new QLabel(" Sample "));
    toolbar->addWidget(m_Sample);
    toolbar->addWidget(new QLabel(" Axis "));
    toolbar->addWidget(m_Axis);
    toolbar->addSeparator();
    toolbar->addWidget(m_Split);
    toolbar->addWidget(m_LogY);
    toolbar->addSeparator();
    toolbar->addWidget(reset);

    connect(reset, &QPushButton::clicked, this, [this]() {
      m_Model->reset();
      sync_rows();
      request_refresh();
    });

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
      r.slider->setValue(tick_of(r.lo, r.hi, info.value));
      // The evaluation is too slow for a live drag on the worst parameters, and
      // inconsistent tracking between rows would be worse than none.
      r.slider->setTracking(false);

      r.spin = new QDoubleSpinBox();
      // Deliberately wider than the slider: typing a value outside the slider's
      // range is a legitimate thing to want, and it simply pins the slider to an
      // end rather than being clamped away.
      r.spin->setRange(r.lo - 1000.0 * std::abs(r.hi - r.lo), r.hi + 1000.0 * std::abs(r.hi - r.lo));
      r.spin->setDecimals(4);
      // The minimiser's own step: arrowing the box once is a step the fit would take.
      r.spin->setSingleStep(info.step > 0.0 ? info.step : (r.hi - r.lo) / 100.0);
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

  void MainWindow::sync_rows() {
    const auto& parameters = m_Model->parameters();

    for (const Row& row : m_Rows) {
      const double value = parameters[static_cast<std::size_t>(row.index)].value;

      // Both widgets are written, so both are blocked: letting either signal
      // through would call back into the model and, worse, fight the other.
      const QSignalBlocker block_slider(row.slider);
      const QSignalBlocker block_spin(row.spin);

      row.slider->setValue(tick_of(row.lo, row.hi, value));
      row.spin->setValue(value);
    }
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
    const int tick = tick_of(row.lo, row.hi, value);

    const QSignalBlocker blocker(row.slider);
    row.slider->setValue(tick);

    m_Model->set(row.index, value);
    request_refresh();
  }

  void MainWindow::refresh() {
    const auto sample = static_cast<std::size_t>(std::max(0, m_Sample->currentIndex()));
    const auto axis   = static_cast<std::size_t>(std::max(0, m_Axis->currentIndex()));

    const double llh          = m_Model->evaluate();
    const auto   marginalized = m_Model->marginalize(sample, axis, m_Split->isChecked());

    const bool   log_y = m_LogY->isChecked();
    const double max_y = max_of(marginalized);
    const double floor = log_y ? log_floor(max_y) : 0.0;

    auto* chart = new QChart();
    chart->setTitle(QString("%1 -- %2").arg(m_Sample->currentText()).arg(m_Axis->currentText()));

    // Overlaid lines, not a stack.
    //
    // A stacked area cannot show what this window exists to show. The prompt
    // atmospheric component is ~0.17% of the conventional one on the tracks
    // sample, so as a band on top of the stack it is sub-pixel: splitting it out
    // changed the legend and nothing else, and driving PromptNorm across its
    // whole range moved the top of the stack by less than a line width. Drawn as
    // its own curve on a log axis it sits three decades down and is perfectly
    // legible, and the slider visibly moves it.
    //
    // The total is drawn separately, so composition is still readable: the gap
    // between the total and the components is what systematicsDelta contributes.
    for (const auto& component : marginalized.components) {
      QLineSeries* series = step_series(marginalized.edges, component.values, floor);
      series->setName(QString::fromStdString(component.name));

      QPen pen(component_color(component.name));
      pen.setWidthF(2.0);
      series->setPen(pen);

      chart->addSeries(series);
    }

    QLineSeries* total = step_series(marginalized.edges, marginalized.total, floor);
    total->setName("total");
    QPen total_pen(QColor(0x33, 0x33, 0x33));
    total_pen.setWidthF(2.5);
    total->setPen(total_pen);
    chart->addSeries(total);

    // Under UseData: false this is the Asimov expectation. At the point that
    // data was generated at -- the AsimovValue, which is not always the start
    // value -- the points land on the total.
    auto* data = new QScatterSeries();
    data->setName("data");
    data->setMarkerSize(5.0);
    data->setColor(Qt::black);
    data->setBorderColor(Qt::black);
    for (std::size_t b = 0; b < marginalized.data.size(); ++b) {
      // An empty bin is left out rather than drawn at the floor. Flooring it
      // puts a marker on the axis line that reads as a real measurement of
      // whatever the floor happens to be, and the tracks sample has a long tail
      // of them; a gap is what an empty bin actually means.
      if (log_y && marginalized.data[b] <= 0.0)
        continue;

      const double centre = 0.5 * (marginalized.edges[b] + marginalized.edges[b + 1]);
      data->append(centre, std::max(marginalized.data[b], floor));
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
      log_axis->setRange(floor, std::max(max_y * 2.0, floor * 10.0));
      // The default format prints a decade like 1e-3 as "0.0", which made the
      // bottom of the axis read as several identical zeroes.
      log_axis->setLabelFormat("%.3g");
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

    chart->legend()->setAlignment(Qt::AlignRight);

    // setChart() takes ownership and deletes the previous chart with all its
    // series, so the ones built above need no separate cleanup.
    m_Chart->setChart(chart);

    m_Status->setText(QString("-2lnL = %1     delta = %2")
                          .arg(llh, 0, 'f', 4)
                          .arg(llh - m_Model->reference_likelihood(), 0, 'f', 4));
  }

}  // namespace explorer
